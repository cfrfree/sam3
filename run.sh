export PYTHONPATH=/root/sam3:$PYTHONPATH
python scripts/train_sam3_refship.py \
  --refer-data-root /root/nas/refship \
  --gpu-ids 0,1 \
  --batch-size 1 \
  --swanlab