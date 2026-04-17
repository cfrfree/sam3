export PYTHONPATH=/root/sam3:$PYTHONPATH
torchrun --nproc_per_node=2 scripts/train_sam3_refship.py \
  --refer-data-root /root/nas/refship \
  --model-id sam3 \
  --swanlab 