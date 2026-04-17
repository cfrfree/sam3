
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 5678 train.py \
--refer_data_root /root/nas/rrsisd/ \
--model_id rrsisd_embedding \
--output-dir /root/nas/rrsisd/logs/rrsis/

# python batch_inference.py