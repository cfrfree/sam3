import argparse


def make_args():
    parser = argparse.ArgumentParser(description="Train SAM3 on RefShip with Referring Segmentation")
    parser.add_argument("--refer-data-root", required=True, help="RefShip dataset root directory")
    parser.add_argument("--dataset", default="refship", help="Dataset name for REFER API")
    parser.add_argument("--split-by", default="unc", help="splitBy for REFER API")
    parser.add_argument("--sam3-checkpoint", default="/root/nas/refship/SAM3/sam3.pt", help="SAM3 checkpoint path if available")
    parser.add_argument("--hf-cache-dir", default=None, help="Directory to cache HuggingFace SAM3 weights")
    parser.add_argument("--img-size", default=480, type=int, help="Resize images to this size")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--output-dir", default="/root/nas/refship/logs/sam3", help="Checkpoint and log directory")
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--experiment-name", default="sam3", help="SwanLab experiment name")
    parser.add_argument("--swanlab", action="store_true", help="Whether to log metrics to SwanLab")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--query-selection",
        default="best_logit",
        choices=["best_logit", "first"],
        help="How to choose the predicted SAM3 query for supervision and evaluation.",
    )
    parser.add_argument("--bce-loss-weight", default=1.0, type=float, help="Weight for BCE mask loss")
    parser.add_argument("--dice-loss-weight", default=1.0, type=float, help="Weight for Dice mask loss")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning for linear layers")
    parser.add_argument("--lora-r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora-groups", default=1, type=int, help="KPLoRA groups (Kronecker groups m)")
    parser.add_argument("--lora-alpha", default=16.0, type=float, help="LoRA alpha (scaling)")
    parser.add_argument("--lora-dropout", default=0.0, type=float, help="LoRA dropout")
    parser.add_argument(
        "--lora-adapter-type",
        default="lora",
        choices=["lora", "dora", "kplora", "moka", "kradapter"],
        help="Adapter type under --use-lora: LoRA, DoRA, KPLoRA, MoKA, or KRAdapter.",
    )
    parser.add_argument(
        "--kplora-share-across-layers",
        action="store_true",
        help="Share KPLoRA A and alpha parameters across layers with the same linear shape.",
    )
    parser.add_argument("--moka-num-experts", default=4, type=int, help="Number of Kronecker experts in MoKA")
    parser.add_argument("--moka-top-k", default=1, type=int, help="Top-k experts activated per token in MoKA")
    parser.add_argument("--lora-k", default=1, type=float, help="LoRA+_lambda")
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
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank for distributed training")
    return parser.parse_args()
