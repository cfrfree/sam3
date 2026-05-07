import argparse
import math
import os
import random
import sys
import time
from dataclasses import fields
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

import swanlab
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import (
	BatchedDatapoint,
	BatchedFindTarget,
	BatchedInferenceMetadata,
	FindStage,
	convert_my_tensors,
)
from rrsis.refer.refer import REFER
import rrsis.transforms as T

# 添加 rrsis 目录到路径，以便导入 rrsis.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rrsis')))
import utils


class RefShipSam3Dataset(Dataset):
	def __init__(
		self,
		refer_root: str,
		split: str = "train",
		dataset: str = "refship",
		split_by: str = "unc",
		img_size: int = 1008,
		eval_mode: bool = False,
	):
		self.refer = REFER(refer_root, dataset, split_by)
		self.split = split
		self.eval_mode = eval_mode
		self.ref_ids = self.refer.getRefIds(split=self.split)
		self.transform = T.Compose(
			[
				T.Resize(img_size, img_size),
				T.ToTensor(),
				T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			]
		)

	def __len__(self):
		return len(self.ref_ids)

	def __getitem__(self, index):
		ref_id = self.ref_ids[index]
		ref = self.refer.loadRefs(ref_id)[0]
		img_info = self.refer.Imgs[ref["image_id"]]
		img_path = os.path.join(self.refer.IMAGE_DIR, img_info["file_name"])
		image = Image.open(img_path).convert("RGB")

		if self.eval_mode:
			sentence = ref["sentences"][0]["raw"]
		else:
			sentence = random.choice(ref["sentences"])["raw"]

		mask_data = self.refer.getMask(ref)["mask"]
		target_mask = Image.fromarray(np.asarray(mask_data, dtype=np.uint8), mode="P")

		if self.transform is not None:
			image, target_mask = self.transform(image, target_mask)

		return image, target_mask, sentence


class LoRALinear(nn.Module):
	"""A minimal LoRA wrapper for nn.Linear."""

	def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
		super().__init__()
		if rank <= 0:
			raise ValueError(f"LoRA rank must be > 0, got {rank}")
		self.base = base_layer
		self.rank = rank
		self.scaling = alpha / rank
		self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
		param_device = base_layer.weight.device
		param_dtype = base_layer.weight.dtype

		self.lora_A = nn.Parameter(
			torch.empty(rank, base_layer.in_features, device=param_device, dtype=param_dtype)
		)
		self.lora_B = nn.Parameter(
			torch.zeros(base_layer.out_features, rank, device=param_device, dtype=param_dtype)
		)
		nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base_out = self.base(x)
		lora_x = self.lora_dropout(x)
		lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
		return base_out + lora_out

	@property
	def weight(self) -> torch.nn.Parameter:
		# Keep compatibility with modules/functions that directly read .weight
		return self.base.weight

	@property
	def bias(self) -> Optional[torch.nn.Parameter]:
		# Keep compatibility with modules/functions that directly read .bias
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
	alpha: float,
	dropout: float,
	target_keywords: List[str],
	exclude_keywords: List[str],
) -> int:
	replaced = 0

	def _inject(module: nn.Module, prefix: str = ""):
		nonlocal replaced
		for child_name, child in list(module.named_children()):
			full_name = f"{prefix}.{child_name}" if prefix else child_name
			# torch.nn.MultiheadAttention may access out_proj.weight/bias directly
			# in functional path; replacing it with LoRA wrapper can break behavior.
			if isinstance(module, nn.MultiheadAttention) and child_name == "out_proj":
				continue
			if isinstance(child, nn.Linear) and _should_apply_lora(full_name, target_keywords, exclude_keywords):
				setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
				replaced += 1
			else:
				_inject(child, full_name)

	_inject(model)
	return replaced


def configure_trainable_params_for_lora(model: nn.Module, train_bias: str = "none", train_norm: bool = False):
	for p in model.parameters():
		p.requires_grad = False

	for name, p in model.named_parameters():
		is_lora_param = name.endswith("lora_A") or name.endswith("lora_B")
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
		if ("lora_A" in k or "lora_B" in k)
	}


def build_box_from_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
	if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
		mask_tensor = mask_tensor.squeeze(0)
	mask_bool = mask_tensor > 0
	if mask_bool.sum() == 0:
		return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

	ys, xs = torch.where(mask_bool)
	y_min = ys.min().item()
	y_max = ys.max().item()
	x_min = xs.min().item()
	x_max = xs.max().item()
	height = mask_tensor.shape[-2]
	width = mask_tensor.shape[-1]
	cx = (x_min + x_max + 1) / 2.0 / width
	cy = (y_min + y_max + 1) / 2.0 / height
	w = (x_max - x_min + 1) / width
	h = (y_max - y_min + 1) / height
	return torch.tensor([cx, cy, w, h], dtype=torch.float32)


def refship_collate(batch):
	images, targets, sentences = zip(*batch)
	batch_size = len(images)
	img_batch = torch.stack(images, dim=0)

	find_stage = FindStage(
		img_ids=list(range(batch_size)),
		text_ids=list(range(batch_size)),
		input_boxes=[torch.zeros(0, 4, dtype=torch.float32) for _ in range(batch_size)],
		input_boxes_label=[torch.zeros(0, dtype=torch.long) for _ in range(batch_size)],
		input_boxes_mask=[torch.ones(0, dtype=torch.bool) for _ in range(batch_size)],
		input_points=[torch.empty(0, 257, dtype=torch.float32) for _ in range(batch_size)],
		input_points_mask=[torch.empty(0, dtype=torch.bool) for _ in range(batch_size)],
		object_ids=[[0] for _ in range(batch_size)],
	)

	boxes = []
	boxes_padded = []
	repeated_boxes = []
	object_ids = []
	object_ids_padded = []
	segments = []
	is_valid_segment = []
	is_exhaustive = []
	original_sizes = []

	for i, target in enumerate(targets):
		target_bool = target.to(torch.bool)
		boxes.append(build_box_from_mask(target_bool))
		boxes_padded.append(build_box_from_mask(target_bool))
		repeated_boxes.append(build_box_from_mask(target_bool))
		object_ids.append([0])
		object_ids_padded.append([0])
		segments.append(target_bool)
		is_valid_segment.append(1)
		is_exhaustive.append(True)
		original_sizes.append(torch.tensor([target.shape[-2], target.shape[-1]], dtype=torch.long))

	find_target = BatchedFindTarget(
		num_boxes=[1] * batch_size,
		boxes=boxes,
		boxes_padded=boxes_padded,
		repeated_boxes=repeated_boxes,
		object_ids=object_ids,
		object_ids_padded=object_ids_padded,
		segments=segments,
		semantic_segments=[],
		is_valid_segment=is_valid_segment,
		is_exhaustive=is_exhaustive,
	)

	metadata = BatchedInferenceMetadata(
		coco_image_id=list(range(batch_size)),
		original_image_id=list(range(batch_size)),
		original_category_id=[0] * batch_size,
		original_size=original_sizes,
		object_id=[0] * batch_size,
		frame_index=[0] * batch_size,
		is_conditioning_only=[False] * batch_size,
	)

	find_stage = convert_my_tensors(find_stage)
	find_target = convert_my_tensors(find_target)
	metadata = convert_my_tensors(metadata)

	return BatchedDatapoint(
		img_batch=img_batch,
		find_text_batch=list(sentences),
		find_inputs=[find_stage],
		find_targets=[find_target],
		find_metadatas=[metadata],
	)


def move_batched_datapoint_to_device(batch: BatchedDatapoint, device: torch.device):
	batch.img_batch = batch.img_batch.to(device, non_blocking=True)

	for stage in batch.find_inputs:
		for field in fields(stage):
			value = getattr(stage, field.name)
			if isinstance(value, torch.Tensor):
				setattr(stage, field.name, value.to(device, non_blocking=True))

	for target in batch.find_targets:
		for field in fields(target):
			value = getattr(target, field.name)
			if isinstance(value, torch.Tensor):
				setattr(target, field.name, value.to(device, non_blocking=True))

	for metadata in batch.find_metadatas:
		for field in fields(metadata):
			value = getattr(metadata, field.name)
			if isinstance(value, torch.Tensor):
				setattr(metadata, field.name, value.to(device, non_blocking=True))

	return batch


def unwrap_model(model: nn.Module) -> nn.Module:
	if isinstance(model, nn.parallel.DistributedDataParallel):
		return model.module
	return model

def evaluate(model, dataloader, device, args):
	model.eval()
	
	# 采用张量以便在多卡间进行 all_reduce 聚合
	total_intersection = torch.tensor(0.0, device=device)
	total_union = torch.tensor(0.0, device=device)
	total_iou_sum = torch.tensor(0.0, device=device)
	total_samples = torch.tensor(0.0, device=device)
	
	thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
	correct = torch.zeros(len(thresholds), dtype=torch.float32, device=device)

	with torch.no_grad():
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			for batch in dataloader:
				batch = move_batched_datapoint_to_device(batch, device)
				outputs = model(batch)
				output = outputs[0]["pred_masks"][:, 0]
				target_mask = batch.find_targets[0].segments
				pred_mask = compute_mask_from_output(output, target_mask.shape)

				for i in range(pred_mask.shape[0]):
					pred = pred_mask[i]
					gt = target_mask[i].to(torch.bool)
					intersection = torch.logical_and(pred, gt).sum().item()
					union = torch.logical_or(pred, gt).sum().item()
					
					iou = 1.0 if union == 0 and intersection == 0 else (intersection / union if union > 0 else 0.0)
					
					total_iou_sum += iou
					total_intersection += intersection
					total_union += union
					for idx, thr in enumerate(thresholds):
						correct[idx] += int(iou >= thr)
					total_samples += 1

	# 【关键修改】在分布式模式下同步所有指标
	if utils.is_dist_avail_and_initialized():
		dist.all_reduce(total_intersection)
		dist.all_reduce(total_union)
		dist.all_reduce(total_iou_sum)
		dist.all_reduce(total_samples)
		dist.all_reduce(correct)

	if total_samples.item() == 0:
		return 0.0, 0.0, {thr: 0.0 for thr in thresholds}

	precision = {thr: 100.0 * float(correct[idx]) / total_samples.item() for idx, thr in enumerate(thresholds)}
	mean_iou = 100.0 * float(total_iou_sum) / total_samples.item()
	overall_iou = 100.0 * float(total_intersection) / float(total_union) if total_union > 0 else 0.0

	return mean_iou, overall_iou, precision


def compute_loss(pred, target):
	pred = pred.unsqueeze(1)  # (2, 1, 288)
	pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
	pred = pred.squeeze(1)  # (2, 1008, 1008)
	return F.binary_cross_entropy_with_logits(pred, target.float())


def compute_mask_from_output(output, shape):
	pred = output.unsqueeze(1)
	pred = F.interpolate(pred, size=shape[-2:], mode='bilinear', align_corners=False)
	pred = pred.squeeze(1)
	# BCEWithLogits uses logits; probability 0.5 corresponds to logit 0.
	return pred > 0

def calculate_dummy_loss(obj):
	"""递归查找所有带有梯度的张量并计算哑损失 (Dummy Loss)"""
	dummy_loss = 0.0
	if isinstance(obj, torch.Tensor):
		if obj.requires_grad:
			dummy_loss += 0.0 * obj.sum()
	elif isinstance(obj, dict):
		for v in obj.values():
			dummy_loss += calculate_dummy_loss(v)
	elif isinstance(obj, (list, tuple)):
		for item in obj:
			dummy_loss += calculate_dummy_loss(item)
	elif hasattr(obj, '__dict__'):  # 处理类似 SAM3Output 等带有属性的自定义对象
		for v in vars(obj).values():
			dummy_loss += calculate_dummy_loss(v)
	elif hasattr(obj, '__iter__') and not isinstance(obj, str): # 兜底其他可迭代对象
		for item in obj:
			dummy_loss += calculate_dummy_loss(item)
	return dummy_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, args):
	model.train()
	
	total_loss = torch.tensor(0.0, device=device)
	count = torch.tensor(0.0, device=device)
	
	start = time.time()
	for batch_idx, batch in enumerate(dataloader):
		batch = move_batched_datapoint_to_device(batch, device)
		optimizer.zero_grad()
		
		# 前向传播
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			outputs = model(batch)
			
			# 1. 计算你的真实 Loss
			output_mask = outputs[0]["pred_masks"][:, 0]  # take the first query's mask
			target_mask = batch.find_targets[0].segments
			loss = compute_loss(output_mask, target_mask)
			
			# 2. 【🌟 终极修复：递归捕获所有深层未使用返回值】
			# 将 outputs 丢进递归函数，它会自动扒出 aux_outputs, prev_encoder_out 等所有梯度张量
			dummy_loss = calculate_dummy_loss(outputs)
			
			# 巧妙地将 dummy_loss 融入计算图，实际不改变 loss 数值
			loss = loss + dummy_loss

		# 反向传播
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		count += 1
		
		# 仅在主进程打印 loss
		if batch_idx % args.print_freq == 0 and utils.is_main_process():
			print(f"Epoch {epoch} | batch {batch_idx}/{len(dataloader)} | loss {loss.item():.4f}")
		
		if utils.is_main_process() and args.swanlab and args.swanlab_enabled:
			swanlab.log({"epoch": epoch, "batch": batch_idx, "loss": loss.item()})

	# 同步 Loss
	if utils.is_dist_avail_and_initialized():
		dist.all_reduce(total_loss)
		dist.all_reduce(count)

	epoch_loss = total_loss.item() / max(1, count.item())
	duration = time.time() - start
	
	if utils.is_main_process():
		print(f"Epoch {epoch} finished. avg_loss={epoch_loss:.4f}. elapsed={duration:.1f}s")
		
	return epoch_loss


def make_args():
	parser = argparse.ArgumentParser(description="Train SAM3 on RefShip with Referring Segmentation")
	parser.add_argument("--refer-data-root", required=True, help="RefShip dataset root directory")
	parser.add_argument("--dataset", default="refship", help="Dataset name for REFER API")
	parser.add_argument("--split-by", default="unc", help="splitBy for REFER API")
	parser.add_argument("--sam3-checkpoint", default="/root/nas/refship/SAM3/sam3.pt", help="SAM3 checkpoint path if available")
	parser.add_argument("--hf-cache-dir", default=None, help="Directory to cache HuggingFace SAM3 weights")
	parser.add_argument("--img-size", default=1008, type=int, help="Resize images to this size")
	parser.add_argument("--batch-size", default=2, type=int)
	parser.add_argument("--workers", default=4, type=int)
	parser.add_argument("--epochs", default=20, type=int)
	parser.add_argument("--lr", default=1e-5, type=float)
	parser.add_argument("--output-dir", default="/root/nas/refship/logs/sam3", help="Checkpoint and log directory")
	parser.add_argument("--print-freq", default=50, type=int)
	parser.add_argument("--experiment-name", default="sam3", help="SwanLab experiment name")
	parser.add_argument("--swanlab", action="store_true", help="Whether to log metrics to SwanLab")
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning for linear layers")
	parser.add_argument("--lora-r", default=8, type=int, help="LoRA rank")
	parser.add_argument("--lora-alpha", default=16.0, type=float, help="LoRA alpha (scaling)")
	parser.add_argument("--lora-dropout", default=0.0, type=float, help="LoRA dropout")
	parser.add_argument(
		"--lora-target-modules",
		default="",
		help="Comma-separated module-name keywords to apply LoRA. Empty means all nn.Linear modules.",
	)
	parser.add_argument(
		"--lora-exclude-modules",
		default="",
		help="Comma-separated module-name keywords to exclude from LoRA.",
	)
	parser.add_argument(
		"--lora-train-bias",
		default="none",
		choices=["none", "all", "lora_only"],
		help="Bias training strategy in LoRA mode.",
	)
	parser.add_argument("--lora-train-norm", action="store_true", help="Whether to train normalization layers in LoRA mode")
	parser.add_argument("--lora-save-only", action="store_true", help="Save LoRA weights only in checkpoints")
	# parser.add_argument('--model-id', default="sam3", type=str, help='Model ID for creating model directory in rrsis utils')
	
	# 【必须添加】分布式训练所需参数
	parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
	
	return parser.parse_args()


def main():
	args = make_args()

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for this training script.")

	# 1. 初始化分布式模式（仅在 torchrun 环境下）
	use_distributed = (
		"RANK" in os.environ
		and "WORLD_SIZE" in os.environ
		and int(os.environ["WORLD_SIZE"]) > 1
	)
	if use_distributed:
		local_rank_env = os.environ.get("LOCAL_RANK")
		args.local_rank = int(local_rank_env) if local_rank_env is not None else int(args.local_rank)
		utils.init_distributed_mode(args)
		rank = utils.get_rank()
		device_id = args.local_rank
	else:
		rank = 0
		device_id = 0
		torch.cuda.set_device(device_id)
	device = torch.device(f"cuda:{device_id}")

	# 固定随机种子保证多卡一致性
	seed = args.seed + rank
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.hf_cache_dir:
		os.environ["HF_HOME"] = args.hf_cache_dir
		os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir

	if utils.is_main_process():
		os.makedirs(args.output_dir, exist_ok=True)
		if args.swanlab:
			swanlab.init(
				project="sam3",
				experiment_name=args.experiment_name,
				config=vars(args),
			)
			args.swanlab_enabled = True
		else:
			args.swanlab_enabled = False

	# 2. 构建 Dataset
	train_dataset = RefShipSam3Dataset(
		refer_root=args.refer_data_root,
		split="train",
		dataset=args.dataset,
		split_by=args.split_by,
		img_size=args.img_size,
		eval_mode=False,
	)
	val_dataset = RefShipSam3Dataset(
		refer_root=args.refer_data_root,
		split="val",
		dataset=args.dataset,
		split_by=args.split_by,
		img_size=args.img_size,
		eval_mode=True, # 修正为 True 保持验证模式一致
	)

	# 3. 构建 DistributedSampler
	if utils.is_dist_avail_and_initialized():
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
	else:
		train_sampler = torch.utils.data.RandomSampler(train_dataset)
		val_sampler = torch.utils.data.SequentialSampler(val_dataset)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=refship_collate,
		drop_last=True # 推荐在多卡训练时丢弃最后的少量batch
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		sampler=val_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=refship_collate,
	)

	# 4. 构建模型并包装为 DDP
	model = build_sam3_image_model(
		checkpoint_path=args.sam3_checkpoint,
		device=device,
		eval_mode=False, # 设置为False进行训练
		load_from_HF=True if args.sam3_checkpoint is None else False,
		enable_segmentation=True,
		image_size=args.img_size,
	)
	model.to(device)

	if args.use_lora:
		target_keywords = [x.strip().lower() for x in args.lora_target_modules.split(",") if x.strip()]
		exclude_keywords = [x.strip().lower() for x in args.lora_exclude_modules.split(",") if x.strip()]
		replaced = inject_lora_modules(
			model,
			rank=args.lora_r,
			alpha=args.lora_alpha,
			dropout=args.lora_dropout,
			target_keywords=target_keywords,
			exclude_keywords=exclude_keywords,
		)
		if replaced == 0:
			raise RuntimeError(
				"LoRA is enabled but no nn.Linear modules were matched. "
				"Please adjust --lora-target-modules / --lora-exclude-modules."
			)
		configure_trainable_params_for_lora(
			model,
			train_bias=args.lora_train_bias,
			train_norm=args.lora_train_norm,
		)
		if utils.is_main_process():
			print(f"LoRA enabled. Replaced linear layers: {replaced}")
	
	if utils.is_dist_avail_and_initialized():
		# 【关键】替换 DataParallel 为 DistributedDataParallel
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = nn.parallel.DistributedDataParallel(
			model, 
			device_ids=[device_id], 
			find_unused_parameters=True
		)

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	if len(trainable_params) == 0:
		raise RuntimeError("No trainable parameters found. Please check training configuration.")
	optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs,  # 总epoch数
    eta_min=args.lr * 0.01  # 最小学习率
	)

	if utils.is_main_process():
		total_p, trainable_p, pct = get_trainable_param_stats(model)
		print(f"Trainable params: {trainable_p:,} / {total_p:,} ({pct:.4f}%)")

	best_miou = float("-inf")
	best_overall_iou = float("-inf")
	best_score = float("-inf")  # score = miou + overall_iou
	best_epoch = -1
	best_precision = {}
	try:
		for epoch in range(1, args.epochs + 1):
			if utils.is_dist_avail_and_initialized() and hasattr(train_sampler, "set_epoch"):
				train_sampler.set_epoch(epoch)

			train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
			lr_scheduler.step()
			val_mean_iou, val_overall_iou, precision = evaluate(model, val_loader, device, args)

			# 5. 仅在主进程保存和日志记录
			if utils.is_main_process():
				print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_mean_iou={val_mean_iou:.2f}% | val_overall_iou={val_overall_iou:.2f}%")
				for thr, prec in precision.items():
					print(f"  precision@{thr:.1f} = {prec:.2f}%")

				if args.swanlab and args.swanlab_enabled:
					swanlab.log({
						"epoch": epoch,
						"train_loss": train_loss,
						"val_mean_iou": val_mean_iou,
						"val_overall_iou": val_overall_iou,
						**{f"precision@{thr}": prec for thr, prec in precision.items()},
					})

				checkpoint = {
					"epoch": epoch,
					"model_state_dict": (
						get_lora_state_dict(unwrap_model(model))
						if args.use_lora and args.lora_save_only
						else unwrap_model(model).state_dict()
					),
					"optimizer_state_dict": optimizer.state_dict(),
					"train_loss": train_loss,
					"val_mean_iou": val_mean_iou,
					"val_overall_iou": val_overall_iou,
				}
				checkpoint_path = os.path.join(args.output_dir, f"sam3_refship_epoch_{epoch}.pth")
				torch.save(checkpoint, checkpoint_path)
				current_score = val_mean_iou + val_overall_iou
				if current_score > best_score:
					best_score = current_score
					best_miou = val_mean_iou
					best_overall_iou = val_overall_iou
					best_epoch = epoch
					best_precision = dict(precision)
					torch.save(checkpoint, os.path.join(args.output_dir, "sam3_refship_best.pth"))
	finally:
		if utils.is_main_process() and best_epoch != -1:
			print(
				f"Global Best @ Epoch {best_epoch} | "
				f"miou={best_miou:.2f}% | overall_iou={best_overall_iou:.2f}%"
			)
			for thr in sorted(best_precision.keys()):
				print(f"  best precision@{thr:.1f} = {best_precision[thr]:.2f}%")

		if utils.is_main_process() and args.swanlab and getattr(args, "swanlab_enabled", False):
			try:
				swanlab.finish()
			except Exception as e:
				print(f"[WARN] SwanLab finish failed: {e}")

		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()


if __name__ == "__main__":
	main()