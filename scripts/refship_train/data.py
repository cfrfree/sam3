import os
import random
from dataclasses import fields

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import rrsis.transforms as T
from rrsis.refer.refer import REFER
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
    convert_my_tensors,
)


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
        refs = self.refer.loadRefs(ref_id)
        if not refs:
            raise RuntimeError(f"Failed to load reference for ref_id={ref_id}")
        ref = refs[0]
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


def build_target_box_from_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
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


def build_text_only_find_stage(batch_size: int) -> FindStage:
    return FindStage(
        img_ids=list(range(batch_size)),
        text_ids=list(range(batch_size)),
        input_boxes=[torch.zeros(0, 4, dtype=torch.float32) for _ in range(batch_size)],
        input_boxes_label=[torch.zeros(0, dtype=torch.long) for _ in range(batch_size)],
        input_boxes_mask=[torch.zeros(0, dtype=torch.bool) for _ in range(batch_size)],
        input_points=[torch.empty(0, 257, dtype=torch.float32) for _ in range(batch_size)],
        input_points_mask=[torch.empty(0, dtype=torch.bool) for _ in range(batch_size)],
        object_ids=[[0] for _ in range(batch_size)],
    )


def refship_collate(batch):
    images, targets, sentences = zip(*batch)
    batch_size = len(images)
    img_batch = torch.stack(images, dim=0)

    find_stage = build_text_only_find_stage(batch_size)

    boxes = []
    boxes_padded = []
    repeated_boxes = []
    object_ids = []
    object_ids_padded = []
    segments = []
    is_valid_segment = []
    is_exhaustive = []
    original_sizes = []

    for target in targets:
        target_bool = target.to(torch.bool)
        target_box = build_target_box_from_mask(target_bool)
        boxes.append(target_box)
        boxes_padded.append(target_box)
        repeated_boxes.append(target_box)
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


def build_dataloaders(args, dist_utils):
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
        eval_mode=True,
    )

    if dist_utils.is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
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
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=refship_collate,
    )
    return train_loader, val_loader, train_sampler
