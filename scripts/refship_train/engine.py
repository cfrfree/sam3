import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

import swanlab

from refship_train.data import move_batched_datapoint_to_device


def _segments_to_tensor(segments):
    if isinstance(segments, torch.Tensor):
        return segments
    return torch.stack([seg for seg in segments], dim=0)


def _select_query_masks(outputs, selection_mode: str):
    pred_masks = outputs["pred_masks"]
    if pred_masks.dim() != 4:
        raise ValueError(f"Expected pred_masks to have shape (B, Q, H, W), got {tuple(pred_masks.shape)}")

    if selection_mode == "first" or pred_masks.shape[1] == 1 or "pred_logits" not in outputs:
        return pred_masks[:, 0]

    pred_logits = outputs["pred_logits"]
    if pred_logits.dim() == 3 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    elif pred_logits.dim() != 2:
        raise ValueError(
            f"Expected pred_logits to have shape (B, Q) or (B, Q, 1), got {tuple(outputs['pred_logits'].shape)}"
        )

    best_query_idx = pred_logits.argmax(dim=1)
    batch_idx = torch.arange(pred_masks.shape[0], device=pred_masks.device)
    return pred_masks[batch_idx, best_query_idx]


def compute_dice_loss(pred, target):
    pred = pred.unsqueeze(1)
    pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.squeeze(1)
    pred = pred.sigmoid()
    target = target.float()
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def compute_loss(pred, target, bce_weight: float, dice_weight: float):
    pred = pred.unsqueeze(1)
    pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.squeeze(1)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target.float())
    dice_loss = compute_dice_loss(pred, target)
    return bce_weight * bce_loss + dice_weight * dice_loss


def compute_mask_from_output(output, shape):
    pred = output.unsqueeze(1)
    pred = F.interpolate(pred, size=shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.squeeze(1)
    return pred > 0


def calculate_dummy_loss(obj):
    dummy_loss = 0.0
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            dummy_loss += 0.0 * obj.sum()
    elif isinstance(obj, dict):
        for v in obj.values():
            dummy_loss += calculate_dummy_loss(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            dummy_loss += calculate_dummy_loss(item)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            dummy_loss += calculate_dummy_loss(v)
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        for item in obj:
            dummy_loss += calculate_dummy_loss(item)
    return dummy_loss


def evaluate(model, dataloader, device, dist_utils, query_selection: str = "best_logit"):
    model.eval()

    total_intersection = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)
    total_iou_sum = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    correct = torch.zeros(len(thresholds), dtype=torch.float32, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in dataloader:
                batch = move_batched_datapoint_to_device(batch, device)
                outputs = model(batch)
                output = _select_query_masks(outputs[0], selection_mode=query_selection)
                target_mask = _segments_to_tensor(batch.find_targets[0].segments)
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

    if dist_utils.is_dist_avail_and_initialized():
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


def train_one_epoch(model, dataloader, optimizer, device, epoch, args, dist_utils):
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        batch = move_batched_datapoint_to_device(batch, device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch)
            output_mask = _select_query_masks(outputs[0], selection_mode=args.query_selection)
            target_mask = _segments_to_tensor(batch.find_targets[0].segments)
            loss = compute_loss(
                output_mask,
                target_mask,
                bce_weight=args.bce_loss_weight,
                dice_weight=args.dice_loss_weight,
            )
            dummy_loss = calculate_dummy_loss(outputs)
            loss = loss + dummy_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if batch_idx % args.print_freq == 0 and dist_utils.is_main_process():
            print(f"Epoch {epoch} | batch {batch_idx}/{len(dataloader)} | loss {loss.item():.4f}")

        if dist_utils.is_main_process() and args.swanlab and args.swanlab_enabled:
            swanlab.log({"epoch": epoch, "batch": batch_idx, "loss": loss.item()})

    if dist_utils.is_dist_avail_and_initialized():
        dist.all_reduce(total_loss)
        dist.all_reduce(count)

    epoch_loss = total_loss.item() / max(1, count.item())
    duration = time.time() - start

    if dist_utils.is_main_process():
        print(f"Epoch {epoch} finished. avg_loss={epoch_loss:.4f}. elapsed={duration:.1f}s")

    return epoch_loss
