import argparse
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


def unwrap_model(model):
	return model.module if hasattr(model, 'module') else model


def compute_mask_from_output(output: dict, target_shape: torch.Size) -> torch.Tensor:
	pred_masks = output["pred_masks"]
	if pred_masks.dim() == 4:
		logits = pred_masks
	elif pred_masks.dim() == 3:
		logits = pred_masks.unsqueeze(1)
	else:
		raise ValueError(f"Unexpected pred_masks dim: {pred_masks.dim()}")

	selected_query = output["pred_logits"].argmax(1).squeeze(1)
	batch_idx = torch.arange(logits.shape[0], device=logits.device)
	best_logits = logits[batch_idx, selected_query]
	if best_logits.shape[-2:] != target_shape[-2:]:
		best_logits = F.interpolate(
			best_logits.unsqueeze(1), size=target_shape[-2:], mode="bilinear", align_corners=False
		).squeeze(1)
	return best_logits.sigmoid() > 0.5


def compute_loss(output: dict, target_mask: torch.Tensor) -> torch.Tensor:
	pred_masks = output["pred_masks"]
	if pred_masks.dim() == 4:
		selected_query = output["pred_logits"].argmax(1).squeeze(1)
		batch_idx = torch.arange(pred_masks.shape[0], device=pred_masks.device)
		logits = pred_masks[batch_idx, selected_query]
	elif pred_masks.dim() == 3:
		logits = pred_masks
	else:
		raise ValueError(f"Unexpected pred_masks dim: {pred_masks.dim()}")

	if logits.shape[-2:] != target_mask.shape[-2:]:
		logits = F.interpolate(
			logits.unsqueeze(1), size=target_mask.shape[-2:], mode="bilinear", align_corners=False
		).squeeze(1)

	target_float = target_mask.to(torch.float32)
	bce = F.binary_cross_entropy_with_logits(logits, target_float)
	pred_probs = torch.sigmoid(logits)
	intersection = (pred_probs * target_float).sum(dim=[1, 2])
	union = pred_probs.sum(dim=[1, 2]) + target_float.sum(dim=[1, 2])
	dice = 1 - (2 * intersection + 1) / (union + 1)
	dice = dice.mean()
	return bce + dice


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
	return (pred > 0.5).float()


def train_one_epoch(model, dataloader, optimizer, device, epoch, args):
	model.train()
	
	total_loss = torch.tensor(0.0, device=device)
	count = torch.tensor(0.0, device=device)
	
	start = time.time()
	for batch_idx, batch in enumerate(dataloader):
		batch = move_batched_datapoint_to_device(batch, device)
		optimizer.zero_grad()
		outputs = model(batch)
		output = outputs[0]["pred_masks"][:, 0]  # take the first query's mask
		target_mask = batch.find_targets[0].segments
		loss = compute_loss(output, target_mask)
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
	parser.add_argument("--experiment-name", default="sam3_refship", help="SwanLab experiment name")
	parser.add_argument("--swanlab", action="store_true", help="Whether to log metrics to SwanLab")
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument('--model-id', default=None, type=str, help='Model ID for creating model directory in rrsis utils')
	
	# 【必须添加】分布式训练所需参数
	parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
	
	return parser.parse_args()


def main():
	args = make_args()
	
	# 1. 初始化多卡分布式模式 (利用 rrsis.utils)
	utils.init_distributed_mode(args)
	rank = utils.get_rank()
	device_id = rank
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
				project="sam3_refship",
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
	)
	model.to(device)
	
	if utils.is_dist_avail_and_initialized():
		# 【关键】替换 DataParallel 为 DistributedDataParallel
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = nn.parallel.DistributedDataParallel(
			model, 
			device_ids=[device_id], 
			find_unused_parameters=True
		)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

	best_val = 0.0
	for epoch in range(1, args.epochs + 1):
		if utils.is_dist_avail_and_initialized():
			train_sampler.set_epoch(epoch)

		train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
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
				"model_state_dict": unwrap_model(model).state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"train_loss": train_loss,
				"val_mean_iou": val_mean_iou,
				"val_overall_iou": val_overall_iou,
			}
			checkpoint_path = os.path.join(args.output_dir, f"sam3_refship_epoch_{epoch}.pth")
			torch.save(checkpoint, checkpoint_path)
			if val_mean_iou > best_val:
				best_val = val_mean_iou
				torch.save(checkpoint, os.path.join(args.output_dir, "sam3_refship_best.pth"))


if __name__ == "__main__":
	main()