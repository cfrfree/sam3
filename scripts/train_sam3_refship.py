import os

import torch
import torch.nn as nn

import swanlab
from sam3.model_builder import build_sam3_image_model

from refship_train.adapters import (
    configure_trainable_params_for_lora,
    get_lora_state_dict,
    get_optimizer_param_groups,
    get_trainable_param_stats,
    inject_lora_modules,
)
from refship_train.args import make_args
from refship_train.common import utils
from refship_train.data import build_dataloaders
from refship_train.engine import evaluate, train_one_epoch
from refship_train.runtime import (
    finalize_run,
    setup_distributed,
    setup_hf_cache,
    setup_logging,
    setup_seed,
    unwrap_model,
)


def main():
    args = make_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device, rank = setup_distributed(args, utils)
    setup_seed(args.seed, rank)
    setup_hf_cache(args.hf_cache_dir)
    setup_logging(args, utils)

    train_loader, val_loader, train_sampler = build_dataloaders(args, utils)

    model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        device=str(device),
        eval_mode=False,
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
            groups=args.lora_groups,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            adapter_type=args.lora_adapter_type,
            kplora_share_across_layers=args.kplora_share_across_layers,
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
            print(f"{args.lora_adapter_type.upper()} enabled. Replaced linear layers: {replaced}")

    if utils.is_dist_avail_and_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        get_optimizer_param_groups(model, base_lr=args.lr, lora_k=args.lora_k),
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    if utils.is_main_process():
        total_p, trainable_p, pct = get_trainable_param_stats(model)
        print(f"Trainable params: {trainable_p:,} / {total_p:,} ({pct:.4f}%)")

    best_miou = float("-inf")
    best_overall_iou = float("-inf")
    best_score = float("-inf")
    best_epoch = -1
    best_precision = {}

    try:
        for epoch in range(1, args.epochs + 1):
            if utils.is_dist_avail_and_initialized():
                set_epoch = getattr(train_sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args, utils)
            lr_scheduler.step()
            val_mean_iou, val_overall_iou, precision = evaluate(model, val_loader, device, utils)

            if utils.is_main_process():
                print(
                    f"Epoch {epoch} | train_loss={train_loss:.4f} | "
                    f"val_mean_iou={val_mean_iou:.2f}% | val_overall_iou={val_overall_iou:.2f}%"
                )
                for thr, prec in precision.items():
                    print(f"  precision@{thr:.1f} = {prec:.2f}%")

                if args.swanlab and args.swanlab_enabled:
                    swanlab.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_mean_iou": val_mean_iou,
                            "val_overall_iou": val_overall_iou,
                            **{f"precision@{thr}": prec for thr, prec in precision.items()},
                        }
                    )

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
        finalize_run(args, best_epoch, best_miou, best_overall_iou, best_precision, utils)


if __name__ == "__main__":
    main()
