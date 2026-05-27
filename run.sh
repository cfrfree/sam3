export PYTHONPATH=/root/sam3:$PYTHONPATH

# Full fine-tuning baseline for RefShip referring segmentation.
# This uses text-only prompts in the dataloader, best-logit query selection,
# and BCE + Dice mask supervision to avoid prompt-side information leakage.
#CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29504 scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 2 \
#   --epochs 40 \
#   --lr 1e-5 \
#   --img-size 480 \
#   --query-selection best_logit \
#   --bce-loss-weight 1.0 \
#   --dice-loss-weight 1.0 \
#   --experiment-name rrsisd_full_ft \
#   --output-dir /root/nas/rrsisd/logs/full_ft \
#   --swanlab


# LoRA baseline
# CUDA_VISIBLE_DEVICES=0 python -u scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 4 \
#   --epochs 40 \
#   --lr 1e-4 \
#   --img-size 480 \
#   --query-selection best_logit \
#   --bce-loss-weight 1.0 \
#   --dice-loss-weight 1.0 \
#   --experiment-name rrsisd_lora \
#   --output-dir /root/nas/rrsisd/logs/lora \
#   --swanlab \
#   --use-lora \
#   --lora-r 32 \
#   --lora-groups 4 \
#   --lora-alpha 64.0 \
#   --lora-dropout 0.05 \
#   --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
#   --lora-train-norm \
#   --lora-save-only \
#   --lora-adapter-type lora


# KPLoRA baseline (no cross-layer sharing)
# CUDA_VISIBLE_DEVICES=1 python -u scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 4 \
#   --epochs 40 \
#   --lr 1e-4 \
#   --img-size 480 \
#   --query-selection best_logit \
#   --bce-loss-weight 1.0 \
#   --dice-loss-weight 1.0 \
#   --experiment-name rrsisd_kplora \
#   --output-dir /root/nas/rrsisd/logs/kplora \
#   --swanlab \
#   --use-lora \
#   --lora-r 32 \
#   --lora-groups 4 \
#   --lora-alpha 64.0 \
#   --lora-dropout 0.05 \
#   --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
#   --lora-train-norm \
#   --lora-save-only \
#   --lora-adapter-type kplora

# KPLoRA + cross-layer sharing (A and alpha shared by same-shape layers)
# Uncomment below to run the shared variant.
# CUDA_VISIBLE_DEVICES=2 python -u scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 4 \
#   --epochs 40 \
#   --lr 1e-4 \
#   --img-size 480 \
#   --query-selection best_logit \
#   --bce-loss-weight 1.0 \
#   --dice-loss-weight 1.0 \
#   --experiment-name rrsisd_kplora_shared \
#   --output-dir /root/nas/rrsisd/logs/kplora_shared \
#   --swanlab \
#   --use-lora \
#   --lora-r 32 \
#   --lora-groups 4 \
#   --lora-alpha 64.0 \
#   --lora-dropout 0.05 \
#   --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
#   --lora-train-norm \
#   --lora-save-only \
#   --lora-adapter-type kplora \
#   --kplora-share-across-layers

# MoKA (sparse MoE over Kronecker experts)
# CUDA_VISIBLE_DEVICES=3 python -u scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 4 \
#   --epochs 40 \
#   --lr 1e-4 \
#   --img-size 480 \
#   --query-selection best_logit \
#   --bce-loss-weight 1.0 \
#   --dice-loss-weight 1.0 \
#   --experiment-name rrsisd_moka \
#   --output-dir /root/nas/rrsisd/logs/moka \
#   --swanlab \
#   --use-lora \
#   --lora-r 32 \
#   --lora-groups 4 \
#   --moka-num-experts 4 \
#   --moka-top-k 1 \
#   --lora-alpha 64.0 \
#   --lora-dropout 0.05 \
#   --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
#   --lora-train-norm \
#   --lora-save-only \
#   --lora-adapter-type moka

# KRAdapter (Khatri-Rao tensor route)
CUDA_VISIBLE_DEVICES=4 python -u scripts/train_sam3_refship.py \
  --refer-data-root /root/nas/rrsisd \
  --batch-size 4 \
  --epochs 40 \
  --lr 1e-4 \
  --img-size 480 \
  --query-selection best_logit \
  --bce-loss-weight 1.0 \
  --dice-loss-weight 1.0 \
  --experiment-name rrsisd_kradapter \
  --output-dir /root/nas/rrsisd/logs/kradapter \
  --swanlab \
  --use-lora \
  --lora-r 32 \
  --lora-groups 4 \
  --lora-alpha 64.0 \
  --lora-dropout 0.05 \
  --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
  --lora-train-norm \
  --lora-save-only \
  --lora-adapter-type kradapter
 