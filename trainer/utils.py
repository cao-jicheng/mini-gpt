import os
import math
import random
import numpy
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: 
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else: 
        Logger(f"Model Params: {total:.2f}M")
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(config, model=None, optimizer=None, epoch=0, step=0, wandb=None, prefix="none", save_dir="../checkpoints", **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = "_moe" if config.use_moe else ''
    ckp_path = f"{save_dir}/{prefix}_{config.hidden_size}_{config.num_hidden_layers}{moe_suffix}.pth"
    # 保存模型checkpoint参数
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)
        ckp_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "wandb_id": wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    ckp_data[key] = raw_value.state_dict()
                else:
                    ckp_data[key] = value
        ckp_tmp = ckp_path + ".tmp"
        torch.save(ckp_data, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        del state_dict, ckp_data
        torch.cuda.empty_cache()
    # 加载模型checkpoint参数
    else:
        if os.path.exists(ckp_path):
            ckp_data = torch.load(ckp_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)