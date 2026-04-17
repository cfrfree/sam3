import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from functools import reduce
import operator
from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils
import numpy as np
import torch.nn.functional as F
import gc
import torch.multiprocessing

# SwanLab 实验记录
import swanlab

torch.multiprocessing.set_sharing_strategy("file_system")


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset

    ds = ReferDataset(
        args, split=image_set, image_transforms=transform, target_transforms=None
    )
    num_classes = 2
    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args, is_train=True):
    if is_train:
        transforms = [
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transforms = [
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return T.Compose(transforms)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).to(input.device)
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, bert_model, device_id, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    all_ious = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            # <<< MODIFIED >>> 使用传入的 device_id
            image = image.to(device_id, non_blocking=True)
            target = target.to(device_id, non_blocking=True)
            sentences = sentences.to(device_id, non_blocking=True)
            attentions = attentions.to(device_id, non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                if getattr(args, 'use_bert_encoder', False):
                    last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                else:
                    embedding = bert_model.module.embeddings(sentences) if hasattr(bert_model, 'module') else bert_model.embeddings(sentences)
                    embedding = embedding.permute(0, 2, 1)
                attentions = attentions.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            all_ious.append(iou)
            cum_I += I
            cum_U += U

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += iou >= eval_seg_iou
            seg_total += 1

    if seg_total == 0:
        if utils.is_main_process():
            print("Warning: Test dataloader is empty, cannot evaluate.")
        return 0.0, 0.0

    mean_iou = np.mean(all_ious)
    overall_iou = cum_I.item() / cum_U.item() if cum_U.item() > 0 else 0.0

    if utils.is_main_process():
        print("Final results:")
        print(f"Mean IoU is {mean_iou * 100.0:.2f}\n")
        results_str = ""
        for n_eval_iou in range(len(eval_seg_iou_list)):
            precision_at_k = seg_correct[n_eval_iou] * 100.0 / seg_total
            results_str += f"    precision@{eval_seg_iou_list[n_eval_iou]} = {precision_at_k:.2f}\n"
        results_str += f"    overall IoU = {overall_iou * 100.0:.2f}\n"
        print(results_str)

    return mean_iou * 100.0, overall_iou * 100.0


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    epoch,
    print_freq,
    iterations,
    bert_model,
    device_id,
    args,
):
    model.train()
    if bert_model is not None:
        bert_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)

    total_loss = 0.0
    num_batches = 0
    for data in metric_logger.log_every(data_loader, print_freq, header):
        iterations += 1
        image, target, sentences, attentions = data

        # <<< MODIFIED >>> 使用传入的 device_id
        image = image.to(device_id, non_blocking=True)
        target = target.to(device_id, non_blocking=True)
        sentences = sentences.to(device_id, non_blocking=True)
        attentions = attentions.to(device_id, non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            if getattr(args, 'use_bert_encoder', False):
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)
            else:
                embedding = bert_model.module.embeddings(sentences) if hasattr(bert_model, 'module') else bert_model.embeddings(sentences)
                embedding = embedding.permute(0, 2, 1)
            attentions = attentions.unsqueeze(dim=-1)
            output = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # SwanLab 记录 batch 级 loss（仅主进程）
        if utils.is_main_process():
            swanlab.log({
                "epoch": epoch,
                "batch": num_batches,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })

        total_loss += loss.item()
        num_batches += 1

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            if 'last_hidden_states' in locals():
                del last_hidden_states
            del embedding
        gc.collect()
        torch.cuda.empty_cache()
    # 返回本 epoch 平均 loss
    return total_loss / max(1, num_batches)


def main(args):
    # 初始化分布式模式
    utils.init_distributed_mode(args)

    # 仅主进程初始化 SwanLab
    if utils.is_main_process():
        swanlab.init(
            project="rrsis",  # 你可以根据需要修改项目名称
            experiment_name=args.model_id,  # 将模型名称作为实验名称
            config=vars(args),  # 自动保存所有传入的命令行参数
        )

    # <<< MODIFIED >>> 获取当前进程的rank作为设备ID
    # 这是最关键的修改，使其不再依赖 args.gpu
    rank = utils.get_rank()
    device_id = rank
    torch.cuda.set_device(device_id)  # 确保后续的.cuda()调用默认使用这个设备

    print(f"Process with rank {rank} is running on device cuda:{device_id}")
    print("Image size: {}".format(str(args.img_size)))

    dataset, num_classes = get_dataset(
        "train", get_transform(args=args, is_train=True), args=args
    )
    dataset_test, _ = get_dataset(
        "val", get_transform(args=args, is_train=False), args=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test, shuffle=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    model = segmentation.__dict__[args.model](
        pretrained=args.pretrained_swin_weights, args=args
    )

    # <<< MODIFIED >>> 将模型移动到当前进程对应的GPU
    model.to(device_id)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # <<< MODIFIED >>> 为DDP指定正确的device_ids
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device_id], find_unused_parameters=True
    )
    single_model = model.module

    if args.model != "lavt_one":
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None

        # <<< MODIFIED >>> 同样地，修改bert_model的设备分配
        bert_model.to(device_id)
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(
            bert_model, device_ids=[device_id], find_unused_parameters=True
        )
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    if args.resume:
        # <<< MODIFIED >>> 使用 map_location 将权重加载到正确的GPU
        map_location = f"cuda:{device_id}"
        checkpoint = torch.load(args.resume, map_location=map_location)
        single_model.load_state_dict(checkpoint["model"])
        if args.model != "lavt_one" and "bert_model" in checkpoint:
            single_bert_model.load_state_dict(checkpoint["bert_model"])

    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if (
            "norm" in name
            or "absolute_pos_embed" in name
            or "relative_position_bias_table" in name
        ):
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != "lavt_one":
        params_to_optimize = [
            {"params": backbone_no_decay, "weight_decay": 0.0},
            {"params": backbone_decay},
            {
                "params": [
                    p for p in single_model.classifier.parameters() if p.requires_grad
                ]
            },
            # the following are the parameters of bert
            {
                "params": reduce(
                    operator.concat,
                    [
                        [
                            p
                            for p in single_bert_model.encoder.layer[i].parameters()
                            if p.requires_grad
                        ]
                        for i in range(10)
                    ],
                )
            },
        ]
    else:
        params_to_optimize = [
            {"params": backbone_no_decay, "weight_decay": 0.0},
            {"params": backbone_decay},
            {
                "params": [
                    p for p in single_model.classifier.parameters() if p.requires_grad
                ]
            },
            # the following are the parameters of bert
            {
                "params": reduce(
                    operator.concat,
                    [
                        [
                            p
                            for p in single_model.text_encoder.encoder.layer[
                                i
                            ].parameters()
                            if p.requires_grad
                        ]
                        for i in range(10)
                    ],
                )
            },
        ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1
    resume_epoch = -1

    if args.resume and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        resume_epoch = checkpoint["epoch"]

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # <<< MODIFIED >>> 传入当前进程的device_id

        train_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            epoch,
            args.print_freq,
            iterations,
            bert_model,
            device_id,
            args,
        )
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, device_id, args)

        if utils.is_main_process():
            print(f"Epoch {epoch}: Mean IoU {iou:.2f}, Overall IoU {overallIoU:.2f}")
            # SwanLab 记录 epoch 级指标
            swanlab.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val/miou": iou,
                "val/oiou": overallIoU,
                "lr": optimizer.param_groups[0]["lr"]
            })
            save_checkpoint = best_oIoU < overallIoU
            if save_checkpoint:
                best_oIoU = overallIoU
                print(
                    f"New best overall IoU: {best_oIoU:.2f}. Saving model for epoch: {epoch}\n"
                )
                dict_to_save = {
                    "model": single_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
                if single_bert_model is not None:
                    dict_to_save["bert_model"] = single_bert_model.state_dict()

                utils.save_on_master(
                    dict_to_save,
                    os.path.join(
                        args.output_dir, "model_best_{}.pth".format(args.model_id)
                    ),
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)
