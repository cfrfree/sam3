import os
import random

import numpy as np
import torch
import torch.distributed as dist

import swanlab


def setup_distributed(args, dist_utils):
    use_distributed = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ["WORLD_SIZE"]) > 1
    )

    if use_distributed:
        local_rank_env = os.environ.get("LOCAL_RANK")
        args.local_rank = int(local_rank_env) if local_rank_env is not None else int(args.local_rank)
        dist_utils.init_distributed_mode(args)
        rank = dist_utils.get_rank()
        device_id = args.local_rank
    else:
        rank = 0
        device_id = 0
        torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}")
    return device, rank


def setup_seed(seed, rank):
    worker_seed = seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_hf_cache(hf_cache_dir):
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir


def setup_logging(args, dist_utils):
    if dist_utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        if args.swanlab:
            swanlab.init(
                project="sam3",
                experiment_name=args.experiment_name,
                config=vars(args),
            )
            args.swanlab_enabled = True
        else:
            args.swanlab_enabled = False
    else:
        args.swanlab_enabled = False


def unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def finalize_run(args, best_epoch, best_miou, best_overall_iou, best_precision, dist_utils):
    if dist_utils.is_main_process() and best_epoch != -1:
        print(
            f"Global Best @ Epoch {best_epoch} | "
            f"miou={best_miou:.2f}% | overall_iou={best_overall_iou:.2f}%"
        )
        for thr in sorted(best_precision.keys()):
            print(f"  best precision@{thr:.1f} = {best_precision[thr]:.2f}%")

    if dist_utils.is_main_process() and args.swanlab and getattr(args, "swanlab_enabled", False):
        try:
            swanlab.finish()
        except Exception as e:
            print(f"[WARN] SwanLab finish failed: {e}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
