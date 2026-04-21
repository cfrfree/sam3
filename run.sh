export PYTHONPATH=/root/sam3:$PYTHONPATH
torchrun --nproc_per_node=2 scripts/train_sam3_refship.py \
  --refer-data-root /root/nas/refship \
  --batch-size 2 \
  --epochs 40 \
  --lr 1e-6 \
  --img-size 480 \
  --model-id sam3 \
  --experiment-name sam3 \
  --output-dir /root/nas/refship/logs/sam3_40e_480 \
  --swanlab 