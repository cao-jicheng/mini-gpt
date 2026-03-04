import os
import re
import time
import argparse
import warnings
import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from model import MiniGPTConfig, MiniGPTForCausalLM
from dataset import RLAIFDataset
from utils import (get_model_params, get_lr, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, Logger, SkipBatchSampler)

warnings.filterwarnings("ignore")

def calculate_rewards(prompts, responses):
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, res, re.S) for res in responses]
        matches_pattern2 = [re.match(pattern2, res, re.S) for res in responses]
        format_rewards = []
        for match_patt, match_patt2 in zip(matches_pattern, matches_pattern2):
            if match_patt or match_patt2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(res) for res in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    _, model_name = os.path.split(args.from_weight)
    if model_name.startswith("reason"):
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0
        for i in range(batch_size):
            for j in range(args.num_generations):
                res_idx = i * args.num_generations + j
                res = responses[res_idx]
                prompt = prompts[i]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                tmp_chat = messages + [{"role": "assistant", "content": res}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -1 * scale)
                if model_name.startswith("reason"):
                    answer_match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -1 * scale)
                        score = score * 0.4 + answer_score * 0.6
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores
    return rewards

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
            padding_side="left", add_special_tokens=False).to(args.device)
        prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -1 * args.max_seq_len:]
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -1 * args.max_seq_len:]
        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(**prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, 
                temperature=0.8, num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
        
        def get_per_token_logps(m, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = m(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            res = model(outputs) if model.config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions).to(args.device)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        per_token_loss = -1 * (torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters - 1:
            current_actor_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            avg_reward = rewards.mean().item()
            avg_res_len = completion_mask.sum(dim=1).float().mean().item()
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"actor_loss: {current_actor_loss:.4f}, aux_loss: {current_aux_loss:.4f}, avg_reward: {avg_reward:.4f}, "
                   f"avg_response_len: {avg_res_len:.2f}, learning_rate: {current_lr:.8f}")
            if wandb and is_main_process():
                wandb.log({
                    "actor_loss": current_actor_loss,
                    "aux_loss": current_aux_loss,
                    "avg_reward": avg_reward,
                    "avg_response_len": avg_res_len,
                    "avg_advantages": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(model.config, model=model, prefix="grpo", optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=step, wandb=wandb)
            model.train()
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif_2000.jsonl", help="强化学习训练数据集")
    parser.add_argument("--reward_model_path", type=str, default="../reward_model/internlm2-1_8b-reward", help="奖励模型路径")
    parser.add_argument("--from_resume", action="store_true", default=False, help="是否从检查点续训（默认不启用）")
    parser.add_argument("--from_weight", type=str, default="../checkpoints/sft_768_16.pth",  help="基于某个权重微调（默认sft）")
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=16, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=66, help="Prompt的最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1000, help="生成文本的最大长度")
    parser.add_argument("--num_generations", type=int, default=8, help="每个Prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--use_moe", action="store_true", default=False, help="是否使用MoE架构（默认不使用）")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每批次训练样本数量")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--save_interval", type=int, default=50, help="模型保存间隔")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="是否使用wandb（默认使用）")
    parser.add_argument("--wandb_project", type=str, default="MiniGPT", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if torch.distributed.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))
    
    # ========== 2. 定义策略模型、参考模型和奖励模型==========
    model = MiniGPTForCausalLM(MiniGPTConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=args.use_moe
    ))
    get_model_params(model, model.config)
    ckp_data = lm_checkpoint(model.config, prefix="grpo", device=args.device) if args.from_resume else None
    if not ckp_data:
        model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"], strict=False)
        Logger(f"Load model weights from {args.from_weight}")
    ref_model = MiniGPTForCausalLM(model.config)
    ref_model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"])
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model = AutoModel.from_pretrained(args.reward_model_path, dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.eval().requires_grad_(False)
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 定义数据和优化器 ==========
    tokenizer = AutoTokenizer.from_pretrained("../model")
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    total_optimizer_steps = (len(loader_for_count) // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume and ckp_data:
        model.load_state_dict(ckp_data["model"])
        scheduler.load_state_dict(ckp_data["scheduler"]) 
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
    reward_model.to(args.device)

    # ========== 6. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"grpo_epoch_{args.epochs}_bs_{args.batch_size}_lr_{args.learning_rate}"
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