import os
import math
import random
import numpy
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler

def get_model_params(model, config):
    # 遍历模型所有参数张量，累加元素个数(numel)，除以 1e6 转换为“百万(M)”单位
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 总共有多少个专家 (Routed Experts)，例如64个
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    # 每个Token实际激活选用的专家数 (Active Experts)，例如每次选2个
    n_active = getattr(config, "num_experts_per_tok", 0)
    # 共享专家数量 (Shared Experts)，这些专家总是被激活
    n_shared = getattr(config, "n_shared_experts", 0)
    # 技巧：通过筛选参数名中包含 'mlp.experts.0.' 的项，只统计“第0号专家”的大小
    # 假设所有专家的结构是一样的，算出一个就能代表所有
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    # 同理，计算单个“共享专家”的大小
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    # 骨架参数 = 总参数 - (单个路由专家 × 总数) - (单个共享专家 × 总数)
    # 这部分包含：Embedding, Attention, RMSNorm, OutputHead等所有非MLP的公共部分
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 激活参数 = 骨架参数 + (单个路由专家 × 激活数) + (单个共享专家 × 总数)
    # 解释：推理时，Token会经过骨架部分，经过所有共享专家，但在路由专家中只走n_active条路
    active = base + (expert * n_active) + (shared_expert * n_shared)
    # 如果激活参数小于总参数，说明这是一个MoE模型
    if active < total: 
        Logger(f"MoE model params: {total:.2f}M-A{active:.2f}M")
    else: 
        Logger(f"Dense model params: {total:.2f}M")
    Logger(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps, lr):
    # 实现了Cosine Annealing策略，学习率会随着step增加呈现余弦曲线下降，通常能带来比固定学习率更好的收敛效果
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

def lm_checkpoint(config, model=None, optimizer=None, epoch=0, step=0, wandb=None, prefix="none", save_dir="../checkpoints", device="cpu", **kwargs):
    # 确保保存目录存在，不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = "_moe" if config.use_moe else ''
    # 自动构造checkpoint的保存路径
    ckp_path = f"{save_dir}/{prefix}_{config.hidden_size}_{config.num_hidden_layers}{moe_suffix}.pth"
    
    # ===== [模式 A：保存Checkpoint] =====
    if model is not None:
        # 如果是DDP模型，取 .module；如果是torch.compile后的模型，取 ._orig_mod
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        # 关键步骤：转为半精度 (half) 并移至CPU，节省磁盘空间且不占用显存
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 获取WandB的run_id，确保重启训练后日志能对接上
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)
        # 构建完整的恢复数据字典
        ckp_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "wandb_id": wandb_id
        }
        # 处理额外的需要保存的对象（如学习率调度器scheduler）
        for key, value in kwargs.items():
            if value is not None:
                # 如果对象有state_dict，保存其状态
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    ckp_data[key] = raw_value.state_dict()
                else:
                    ckp_data[key] = value
        # 安全保存模型权重 (使用.tmp中转，防止保存过程中途崩溃，导致文件损坏)
        ckp_tmp = ckp_path + ".tmp"
        torch.save(ckp_data, ckp_tmp)
        # 写完后再通过系统级指令瞬间改名覆盖
        os.replace(ckp_tmp, ckp_path)
        # 显存清理
        del state_dict, ckp_data
        torch.cuda.empty_cache()

    # ===== [模式 B：加载Checkpoint] =====
    else:
        if os.path.exists(ckp_path):
            ckp_data = torch.load(ckp_path, map_location=device)
            # 处理GPU数量变化后的Step转换
            # 例如：之前用2张卡跑了100 step，现在换成4张卡，
            # 为了保持数据消耗量一致，step需要调整（100 * 2 // 4 = 50）
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        """初始化跳过批次的采样器
        :param sampler: 基础采样器（如 SequentialSampler 或 DistributedSampler），决定了索引的原始顺序
        :param batch_size: 批大小
        :param skip_batches: 需要跳过的Batch数量（通常从checkpoint中读取）
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        # 用于暂存当前Batch的索引
        batch = []
        # 记录已经跳过的Batch计数器
        skipped = 0
        # 遍历基础采样器产生的所有样本索引
        for idx in self.sampler:
            batch.append(idx)
            # 当收集的索引数量达到一个Batch大小时
            if len(batch) == self.batch_size:
                # 检查是否还需要继续跳过
                if skipped < self.skip_batches:
                    skipped += 1
                    # 清空当前Batch，但不返回给DataLoader
                    batch = []
                    continue
                # 如果已经跳够了，则产出该Batch
                yield batch
                # 重置Batch准备收集下一个
                batch = []
        # 如果样本总数不能被batch_size整除，处理最后剩下的不足一个Batch的样本
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # 1. 计算原始数据总共能分成多少个Batch (向上取整)
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        # 2. 返回剩余的Batch数量，确保不为负数
        return max(0, total_batches - self.skip_batches)