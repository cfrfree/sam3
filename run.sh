export PYTHONPATH=/root/sam3:$PYTHONPATH

# KPLoRA baseline (no cross-layer sharing)
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29504 scripts/train_sam3_refship.py \
#   --refer-data-root /root/nas/rrsisd \
#   --batch-size 2 \
#   --epochs 40 \
#   --lr 1e-4 \
#   --img-size 480 \
#   --experiment-name sam3_rrsisd_40e_480_kplora \
#   --output-dir /root/nas/rrsisd/logs/sam3_40e_480_kplora \
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
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 scripts/train_sam3_refship.py \
  --refer-data-root /root/nas/rrsisd \
  --batch-size 2 \
  --epochs 40 \
  --lr 1e-4 \
  --img-size 480 \
  --experiment-name sam3_rrsisd_40e_480_kplora_shared \
  --output-dir /root/nas/rrsisd/logs/sam3_40e_480_kplora_shared \
  --swanlab \
  --use-lora \
  --lora-r 32 \
  --lora-groups 4 \
  --lora-alpha 64.0 \
  --lora-dropout 0.05 \
  --lora-target-modules "transformer.decoder.layers.ca_text,transformer.decoder.layers.cross_attn,backbone.vision_backbone.trunk.blocks.16.attn.qkv,backbone.vision_backbone.trunk.blocks.17.attn.qkv,backbone.vision_backbone.trunk.blocks.18.attn.qkv,backbone.vision_backbone.trunk.blocks.19.attn.qkv,backbone.vision_backbone.trunk.blocks.20.attn.qkv,backbone.vision_backbone.trunk.blocks.21.attn.qkv,backbone.vision_backbone.trunk.blocks.22.attn.qkv,backbone.vision_backbone.trunk.blocks.23.attn.qkv,backbone.vision_backbone.trunk.blocks.24.attn.qkv,backbone.vision_backbone.trunk.blocks.25.attn.qkv,backbone.vision_backbone.trunk.blocks.26.attn.qkv,backbone.vision_backbone.trunk.blocks.27.attn.qkv,backbone.vision_backbone.trunk.blocks.28.attn.qkv,backbone.vision_backbone.trunk.blocks.29.attn.qkv,backbone.vision_backbone.trunk.blocks.30.attn.qkv,backbone.vision_backbone.trunk.blocks.31.attn.qkv,backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj,backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj,backbone.language_backbone.resizer" \
  --lora-train-norm \
  --lora-save-only \
  --lora-adapter-type kplora \
  --kplora-share-across-layers
 