import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KPLoRALinear(nn.Module):
    """
    A KPLORA wrapper for nn.Linear based on the PERT-RaMa paper.
    Requires in_features and out_features to be divisible by groups.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        groups: int,
        alpha: float,
        dropout: float,
        shared_A: Optional[nn.Parameter] = None,
        shared_lora_alpha_param: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        if rank <= 0 or groups <= 0:
            raise ValueError(f"Rank and groups must be > 0, got rank={rank}, groups={groups}")

        self.base = base_layer
        self.r = rank
        self.m = groups
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        if in_f % self.m != 0 or out_f % self.m != 0:
            raise ValueError(
                f"in_features ({in_f}) and out_features ({out_f}) must be divisible by groups (m={self.m})"
            )

        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        expected_a_shape = (self.m, self.m, self.m)
        expected_alpha_shape = (self.m, in_f // self.m, self.r)

        # A_i in R^{m x m}, total m groups -> shape (m, m, m)
        if shared_A is None:
            self.A = nn.Parameter(torch.empty(*expected_a_shape, device=param_device, dtype=param_dtype))
            nn.init.xavier_normal_(self.A)
        else:
            if tuple(shared_A.shape) != expected_a_shape:
                raise ValueError(
                    f"shared_A shape mismatch. Expected {expected_a_shape}, got {tuple(shared_A.shape)}"
                )
            self.A = shared_A

        # alpha_i in R^{(Din/m) x r}, total m groups -> shape (m, Din/m, r)
        if shared_lora_alpha_param is None:
            self.lora_alpha_param = nn.Parameter(
                torch.empty(*expected_alpha_shape, device=param_device, dtype=param_dtype)
            )
            nn.init.xavier_normal_(self.lora_alpha_param)
        else:
            if tuple(shared_lora_alpha_param.shape) != expected_alpha_shape:
                raise ValueError(
                    "shared_lora_alpha_param shape mismatch. "
                    f"Expected {expected_alpha_shape}, got {tuple(shared_lora_alpha_param.shape)}"
                )
            self.lora_alpha_param = shared_lora_alpha_param

        # beta_i in R^{r x (Dout/m)}, total m groups -> shape (m, r, Dout/m)
        self.lora_beta_param = nn.Parameter(
            torch.zeros(self.m, self.r, out_f // self.m, device=param_device, dtype=param_dtype)
        )

        # PERT-RaMa ablation: beta uses zero init.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = self.lora_dropout(x)

        # Delta W has shape (D_in, D_out) to align with F.linear input convention after transpose.
        delta_w = torch.zeros(self.in_features, self.out_features, device=x.device, dtype=lora_x.dtype)

        for i in range(self.m):
            # B_i = alpha_i @ beta_i -> shape (D_in/m, D_out/m)
            b_i = torch.matmul(self.lora_alpha_param[i], self.lora_beta_param[i])
            # Kronecker product -> shape (D_in, D_out)
            delta_w = delta_w + torch.kron(self.A[i], b_i).to(dtype=delta_w.dtype)

        # F.linear expects weight in (out_features, in_features).
        lora_out = F.linear(lora_x, delta_w.t()) * self.scaling
        return base_out + lora_out

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


class MoKALinear(nn.Module):
    """
    Mixture of Kronecker Product Adaptation (MoKA).
    Uses token-wise sparse routing over Kronecker experts.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        groups: int,
        num_experts: int,
        top_k: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0 or groups <= 0 or num_experts <= 0:
            raise ValueError(
                f"Rank/groups/num_experts must be > 0, got rank={rank}, groups={groups}, num_experts={num_experts}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got top_k={top_k}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be larger than num_experts ({num_experts})")

        self.base = base_layer
        self.r = rank
        self.m = groups
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        if in_f % self.m != 0 or out_f % self.m != 0:
            raise ValueError(
                f"in_features ({in_f}) and out_features ({out_f}) must be divisible by groups (m={self.m})"
            )

        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        self.lora_A_experts = nn.Parameter(
            torch.empty(self.num_experts, self.m, self.m, device=param_device, dtype=param_dtype)
        )
        self.lora_alpha_experts = nn.Parameter(
            torch.empty(self.num_experts, in_f // self.m, self.r, device=param_device, dtype=param_dtype)
        )
        self.lora_beta_experts = nn.Parameter(
            torch.zeros(self.num_experts, self.r, out_f // self.m, device=param_device, dtype=param_dtype)
        )
        self.lora_router = nn.Linear(in_f, self.num_experts, bias=False, device=param_device, dtype=param_dtype)

        nn.init.xavier_normal_(self.lora_A_experts)
        nn.init.xavier_normal_(self.lora_alpha_experts)
        nn.init.xavier_uniform_(self.lora_router.weight)

    def _expert_delta_w(self, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
        b_i = torch.matmul(self.lora_alpha_experts[expert_idx], self.lora_beta_experts[expert_idx])
        return torch.kron(self.lora_A_experts[expert_idx], b_i).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = self.lora_dropout(x)

        x_shape = lora_x.shape
        x_flat = lora_x.reshape(-1, self.in_features)

        gate_logits = self.lora_router(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        sparse_gates = torch.zeros_like(gate_probs)
        sparse_gates.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        sparse_gates = sparse_gates / sparse_gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        lora_out_flat = torch.zeros(
            x_flat.shape[0],
            self.out_features,
            device=x_flat.device,
            dtype=x_flat.dtype,
        )

        for expert_idx in range(self.num_experts):
            token_mask = sparse_gates[:, expert_idx] > 0
            if not torch.any(token_mask):
                continue
            delta_w = self._expert_delta_w(expert_idx, dtype=x_flat.dtype)
            expert_out = F.linear(x_flat[token_mask], delta_w.t())
            gate_weight = sparse_gates[token_mask, expert_idx].unsqueeze(-1)
            lora_out_flat[token_mask] = lora_out_flat[token_mask] + gate_weight * expert_out

        lora_out = lora_out_flat.reshape(*x_shape[:-1], self.out_features) * self.scaling
        return base_out + lora_out

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


class KRAdapterLinear(nn.Module):
    """
    KRAdapter based on Khatri-Rao product parameterization:
    DeltaW = (U * V) @ Wm.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, groups: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0 or groups <= 0:
            raise ValueError(f"Rank and groups must be > 0, got rank={rank}, groups={groups}")

        self.base = base_layer
        self.rank = rank
        self.groups = groups
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        if in_f % self.groups != 0:
            raise ValueError(f"in_features ({in_f}) must be divisible by groups ({self.groups}) for KRAdapter")

        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        # (groups, rank) and (in_f/groups, rank) -> Khatri-Rao gives (in_f, rank)
        self.lora_U = nn.Parameter(torch.empty(self.groups, self.rank, device=param_device, dtype=param_dtype))
        self.lora_V = nn.Parameter(
            torch.empty(in_f // self.groups, self.rank, device=param_device, dtype=param_dtype)
        )
        self.lora_Wm = nn.Parameter(torch.zeros(self.rank, out_f, device=param_device, dtype=param_dtype))

        nn.init.xavier_normal_(self.lora_U)
        nn.init.xavier_normal_(self.lora_V)

    def _khatri_rao(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Column-wise Kronecker product: (a, r) * (b, r) -> (a*b, r)
        a, r_u = u.shape
        b, r_v = v.shape
        if r_u != r_v:
            raise ValueError(f"Khatri-Rao requires same rank, got {r_u} and {r_v}")
        return torch.einsum("ar,br->abr", u, v).reshape(a * b, r_u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = self.lora_dropout(x)

        kr_mat = self._khatri_rao(self.lora_U, self.lora_V).to(dtype=lora_x.dtype)
        delta_w = torch.matmul(kr_mat, self.lora_Wm.to(dtype=lora_x.dtype))
        lora_out = F.linear(lora_x, delta_w.t()) * self.scaling
        return base_out + lora_out

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


class LoRALinear(nn.Module):
    """LoRA/DoRA wrapper for nn.Linear, controlled by adapter_type."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        adapter_type: str = "dora",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if adapter_type not in {"lora", "dora"}:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}. Expected 'lora' or 'dora'.")

        self.base = base_layer
        self.adapter_type = adapter_type
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout_p = float(dropout)
        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=param_device, dtype=param_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=param_device, dtype=param_dtype)
        )
        if self.adapter_type == "dora":
            self.dora_magnitude = nn.Parameter(
                base_layer.weight.detach().norm(p=2, dim=1, keepdim=True).to(device=param_device, dtype=param_dtype)
            )
        else:
            self.register_parameter("dora_magnitude", None)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_type == "lora":
            base_out = self.base(x)
            lora_x = F.dropout(x, p=self.lora_dropout_p, training=self.training) if self.lora_dropout_p > 0 else x
            lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        if self.training and self.lora_dropout_p > 0:
            delta_w = F.dropout(delta_w, p=self.lora_dropout_p, training=True)

        combined_w = self.base.weight + delta_w
        combined_norm = combined_w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        direction_w = combined_w / combined_norm
        dora_w = self.dora_magnitude * direction_w
        return F.linear(x, dora_w, self.base.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _should_apply_lora(module_name: str, target_keywords: List[str], exclude_keywords: List[str]) -> bool:
    lower_name = module_name.lower()
    if exclude_keywords and any(k in lower_name for k in exclude_keywords):
        return False
    if not target_keywords:
        return True
    return any(k in lower_name for k in target_keywords)


def inject_lora_modules(
    model: nn.Module,
    rank: int,
    groups: int,
    moka_num_experts: int,
    moka_top_k: int,
    alpha: float,
    dropout: float,
    adapter_type: str,
    kplora_share_across_layers: bool,
    target_keywords: List[str],
    exclude_keywords: List[str],
) -> int:
    replaced = 0
    shared_kplora_params = {}

    def _inject(module: nn.Module, prefix: str = ""):
        nonlocal replaced
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(module, nn.MultiheadAttention) and child_name == "out_proj":
                continue
            if isinstance(child, nn.Linear) and _should_apply_lora(full_name, target_keywords, exclude_keywords):
                replacement_layer: nn.Module
                if adapter_type == "kplora":
                    if kplora_share_across_layers:
                        share_key = (child.in_features, child.out_features, groups, rank)
                        shared = shared_kplora_params.get(share_key)
                        if shared is None:
                            replacement_layer = KPLoRALinear(
                                child,
                                rank=rank,
                                groups=groups,
                                alpha=alpha,
                                dropout=dropout,
                            )
                            shared_kplora_params[share_key] = (
                                replacement_layer.A,
                                replacement_layer.lora_alpha_param,
                            )
                        else:
                            shared_A, shared_alpha = shared
                            replacement_layer = KPLoRALinear(
                                child,
                                rank=rank,
                                groups=groups,
                                alpha=alpha,
                                dropout=dropout,
                                shared_A=shared_A,
                                shared_lora_alpha_param=shared_alpha,
                            )
                    else:
                        replacement_layer = KPLoRALinear(
                            child,
                            rank=rank,
                            groups=groups,
                            alpha=alpha,
                            dropout=dropout,
                        )
                elif adapter_type == "moka":
                    replacement_layer = MoKALinear(
                        child,
                        rank=rank,
                        groups=groups,
                        num_experts=moka_num_experts,
                        top_k=moka_top_k,
                        alpha=alpha,
                        dropout=dropout,
                    )
                elif adapter_type == "kradapter":
                    replacement_layer = KRAdapterLinear(
                        child,
                        rank=rank,
                        groups=groups,
                        alpha=alpha,
                        dropout=dropout,
                    )
                else:
                    replacement_layer = LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        adapter_type=adapter_type,
                    )
                setattr(
                    module,
                    child_name,
                    replacement_layer,
                )
                replaced += 1
            else:
                _inject(child, full_name)

    _inject(model)
    return replaced


def configure_trainable_params_for_lora(model: nn.Module, train_bias: str = "none", train_norm: bool = False):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        is_lora_param = ("lora_" in name or name.endswith(".A") or name.endswith("dora_magnitude"))
        if is_lora_param:
            p.requires_grad = True
            continue

        if train_bias == "all" and name.endswith("bias"):
            p.requires_grad = True
        elif train_bias == "lora_only" and name.endswith("base.bias"):
            p.requires_grad = True

    if train_norm:
        for m in model.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                for p in m.parameters():
                    p.requires_grad = True


def get_trainable_param_stats(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return total, trainable, pct


def get_lora_state_dict(model: nn.Module):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ("lora_" in k or k.endswith(".A") or "dora_magnitude" in k)
    }


def get_optimizer_param_groups(model: nn.Module, base_lr: float, lora_k: float):
    lora_a_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and (
            "lora_A" in n
            or "lora_alpha_param" in n
            or "lora_alpha_experts" in n
            or "lora_U" in n
            or "lora_V" in n
            or "lora_router" in n
            or n.endswith(".A")
        )
    ]
    lora_b_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and ("lora_B" in n or "lora_beta_param" in n or "lora_beta_experts" in n or "lora_Wm" in n)
    ]
    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and (
            "lora_A" not in n
            and "lora_B" not in n
            and "lora_alpha_param" not in n
            and "lora_beta_param" not in n
            and "lora_alpha_experts" not in n
            and "lora_beta_experts" not in n
            and "lora_U" not in n
            and "lora_V" not in n
            and "lora_Wm" not in n
            and "lora_router" not in n
            and not n.endswith(".A")
        )
    ]
    return [
        {"params": lora_a_params, "lr": base_lr},
        {"params": lora_b_params, "lr": base_lr * lora_k},
        {"params": other_params, "lr": base_lr},
    ]
