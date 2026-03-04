import os
import time
import argparse
import warnings
import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model import MiniGPTConfig, MiniGPTForCausalLM
from dataset import DPODataset
from utils import (get_model_params, get_lr, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, Logger, SkipBatchSampler)

warnings.filterwarnings("ignore")

def logits_to_log_probs(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token

def get_dpo_loss(ref_log_probs, actor_log_probs, mask, beta):
    seq_length = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_length.squeeze()
    actor_log_probs = (actor_log_probs * mask).sum(dim=1) / seq_length.squeeze()
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_actor_log_probs = actor_log_probs[:batch_size // 2]
    reject_actor_log_probs = actor_log_probs[batch_size // 2:]
    pi_logratios = chosen_actor_log_probs - reject_actor_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -1 * torch.nn.functional.logsigmoid(beta * logits)
    return loss.mean()

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            actor_outputs = model(x)
            actor_logits = actor_outputs.logits
            actor_log_probs = logits_to_log_probs(actor_logits, y)

            dpo_loss = get_dpo_loss(ref_log_probs, actor_log_probs, mask, beta=args.beta)
            loss = dpo_loss + actor_outputs.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss.item()
            current_aux_loss = actor_outputs.aux_loss.item() if actor_outputs.aux_loss is not None else 0.0
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, "
                   f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta_time: {eta_min:.1f} min")
            if wandb: 
                wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, 
                    "learning_rate": current_lr, "eta_time (minutes)": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(model.config, model=model, prefix="dpo", optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb)
            model.train()
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, actor_outputs, actor_logits, actor_log_probs, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT DPO (Direct Preference Optimization)")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="强化学习训练数据集")
    parser.add_argument("--from_weight", type=str, default="../checkpoints/sft_768_16.pth",  help="基于某个权重微调（默认sft）")
    parser.add_argument("--from_resume", action="store_true", default=False, help="是否从检查点续训（默认不启用）")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO中的beta参数")
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=16, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--use_moe", action="store_true", default=False, help="是否使用MoE架构（默认不使用）")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每批次训练样本数量")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="是否使用wandb（默认使用）")
    parser.add_argument("--wandb_project", type=str, default="MiniGPT", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if torch.distributed.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))
    
    # ========== 2. 定义模型、参考模型 ==========
    model = MiniGPTForCausalLM(MiniGPTConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=args.use_moe
    ))
    get_model_params(model, model.config)
    ckp_data = lm_checkpoint(model.config, prefix="dpo", device=args.device) if args.from_resume else None
    if not ckp_data:
        model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"], strict=False)
        Logger(f"Load model weights from {args.from_weight}")
    ref_model = MiniGPTForCausalLM(model.config)
    ref_model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"])
    ref_model.eval().requires_grad_(False)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 定义数据和优化器 ==========
    tokenizer = AutoTokenizer.from_pretrained("../model")
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume and ckp_data:
        model.load_state_dict(ckp_data["model"])
        scaler.load_state_dict(ckp_data["scaler"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        # 需要手动切换optimizer中tensor的训练设备
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
    model.to(args.device)
    ref_model.to(args.device)
    
    # ========== 6. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"dpo_epoch_{args.epochs}_bs_{args.batch_size}_lr_{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 7. DDP包模型 ==========
    if torch.distributed.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step, 从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if torch.distributed.is_initialized(): 
        torch.distributed.destroy_process_group()