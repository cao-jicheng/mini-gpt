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
    """整合所有奖励函数，计算总奖励
    prompts: [batch_size]
    responses: [batch_size * num_gen]
    """ 
    def reasoning_model_reward(rewards):
        """基于回答格式和标签规则的启发式奖励函数 (Rule-based Reward)"""
        # 正则表达式：严格匹配以 <think> 开始和结束，接着以 <answer> 开始和结束的格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 兼容 <think> 和 <answer> 之间有一个空行的格式
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        # 检查每个生成的response是否完全符合上述两种格式之一
        matches_pattern = [re.match(pattern, res, re.S) for res in responses]
        matches_pattern2 = [re.match(pattern2, res, re.S) for res in responses]
        format_rewards = []
        for match_patt, match_patt2 in zip(matches_pattern, matches_pattern2):
            # 如果格式完全匹配，给予0.5的基础格式奖励
            if match_patt or match_patt2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        # 将格式奖励叠加到总奖励张量上
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            """统计关键标签的出现次数，每出现一次正确标签给予额外奖励"""
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward
        # 计算批次中每个response的标签数量奖励
        mark_rewards = [mark_num(res) for res in responses]
        # 将标签奖励叠加到总奖励张量上
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards
    # 初始化一个全 0 的张量，用于存储每个response的最终得分
    rewards = torch.zeros(len(responses), device=args.device)
    # 根据传入的模型权重名称，自动推断是否是推理模型
    _, model_name = os.path.split(args.from_weight)
    if model_name.startswith("reason"):
        # 如果是推理模型，优先计算格式相关的启发式奖励
        rewards = reasoning_model_reward(rewards)
    # 外部奖励模型前向计算，禁用梯度计算
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        # 设置奖励模型的截断范围为[-3.0, 3.0]
        scale = 3.0
        # 遍历批次中的每个prompt
        for i in range(batch_size):
            # 遍历针对该prompt生成的num_generations个不同的response
            for j in range(args.num_generations):
                res_idx = i * args.num_generations + j
                res = responses[res_idx]
                prompt = prompts[i]
                # 解析prompt，将其从纯文本转换为标准的ChatML消息列表格式
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                # 将模型生成的response作为assistant角色追加到消息列表中
                tmp_chat = messages + [{"role": "assistant", "content": res}]
                # 调用外部奖励模型评估整个对话，得到一个打分
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # 将分数限制在设定好的scale范围内，防止奖励模型偶尔“发疯”，给出极端值导致训练崩溃
                score = max(min(score, scale), -1 * scale)
                # 如果是推理模型，还需要对 <answer> 标签内的最终答案进行单独评分
                if model_name.startswith("reason"):
                    # 提取 <answer> 和 </answer> 之间的内容
                    answer_match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # 构建仅包含最终答案的对话进行评分
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -1 * scale)
                        # 综合得分：整个回复得分占40%，核心答案得分占60%
                        # 这个设计使得，整个包含废话和思考过程的回复只占40%的权重，
                        # 而最终结论的正确与否占据了60%的主导地位。
                        # 这能逼迫模型把注意力放在“得出正确结论”上，而不是写一堆看似合理但毫无用处的思考过程。
                        score = score * 0.4 + answer_score * 0.6
                reward_model_scores.append(score)
        # 将列表转换为张量并加到总rewards上
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores
    # rewards形状为 [batch_size * num_gen]，与responses对应
    return rewards

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        # 注意这里使用的是padding_side="left"。因为在做因果语言模型（Causal LM）的自回归生成时，
        # 所有的生成都是接在最右侧的，所以左侧填充能保证右侧是对齐的。
        # 这里也与RLAIFDataset中没有token化相对应，在这里才进行token化。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
            padding_side="left", add_special_tokens=False).to(args.device)
        # 从左侧截断（保留最新内容，为右侧生成做准备）
        prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -1 * args.max_seq_len:]
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -1 * args.max_seq_len:]
        # 纯生成阶段，不需要计算梯度，节省大量显存
        with torch.no_grad():
            # DDP模型需要使用.module访问generate方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # 让模型为每个prompt生成num_generations个回答
            outputs = model_for_gen.generate(**prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, 
                temperature=0.8, num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)
        # outputs包含了prompt+生成的回答，我们把生成的回答部分单独切出来
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
        
        def get_per_token_logps(mdl, input_ids, n_keep):
            """给定token序列，计算模型生成这些token的对数概率
            input_ids: [batch_size * num_gen, P+R]，P是Prompt长度，R是生成的响应长度
            n_keep: 需要保留的token数量（即生成的回答部分）
            """
            # 如果input_ids是推理模式，则需要detach克隆
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # logits形状为 [batch_size * num_gen, P+R, vocab_size]
            # 错位切片 [:, :-1, :]，这是一个极其关键的自回归操作！
            # 因为语言模型是用第t个位置的特征去预测第t+1个位置的词，
            # 所以我们要把Logits的最后一个位置丢掉（因为它预测的是序列外未知的下一个词），
            # 这样切片后的Logits矩阵正好和我们要评估的目标token序列在位置上一一对应了。
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            # per_token_logps形状为 [batch_size * num_gen, R]
            per_token_logps = []
            # ids_row是模型实际生成的Token ID，截取最后n_keep个
            for logits_row, ids_row in zip(logits, input_ids[:, -1 * n_keep:]):
                # 如果ids_row是推理模式，则需要detach克隆
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 1. log_softmax先把Logits转为对数概率
                # 2. gather操作从巨大的词表概率中，精准“抠”出实际生成的那个字的概率
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            # 1. 计算当前策略模型生成这些结果的对数概率 (开启梯度计算，per_token_logps是后续梯度下降求导的源头)
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            # MOE模型需要再做一次前向计算，以获取辅助损失
            res = model(outputs) if model.config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        with torch.no_grad():
            # 2. 计算参考模型生成一模一样结果的对数概率 (不计算梯度，作为锚点)
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))
        
        # Q：为什么同样的东西要算两遍（两次运行get_per_token_logps）？
        # A：因为我们需要防止策略模型为了拿高分而“走火入魔”（比如发现奖励模型喜欢感叹号，就通篇输出感叹号）。
        # 我们算出ref_per_token_logps（老模型原本是怎么说话的），再算出per_token_logps（新模型现在是怎么说话的），
        # 两者相减就可以计算出KL散度 (KL Divergence)。KL散度越大约等于惩罚越重，从而强迫新模型在提高分数的同时，
        # 依然保持老模型原本的语言逻辑和流利度。

        # 把token ID解码回文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # 用calculate_rewards裁判函数，给所有的回答打分
        rewards = calculate_rewards(prompts, completions).to(args.device)

        # ===== GRPO 组内优势计算 =====
        grouped_rewards = rewards.view(-1, args.num_generations)
        # 组内平均分和标准差
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        # Advantage = (奖励 - 组内均值) / 组内标准差
        # .clamp截断防止梯度爆炸
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        # advantages归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 计算结束符 (EOS) 后的Mask，忽略掉填充的无意义字符
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()
        # 计算参考模型与策略模型的KL散度
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # GRPO Loss核心计算公式
        # torch.exp(per_token_logps - per_token_logps.detach()) 相当于重要性采样比率 (ratio)
        # 乘以优势(advantages)，再减去KL惩罚(args.beta * per_token_kl)
        per_token_loss = -1 * (torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)
        # 利用Mask把无效token的损失过滤掉，求序列的平均Loss
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # 加上MoE的辅助Loss (如果有)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        # 反向传播求梯度
        loss.backward()
        # 梯度更新 (Optimizer Step) - 仅在达到累积步数时执行
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪，避免训练崩溃
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 模型权重更新
            optimizer.step()
            # 学习率衰减
            scheduler.step()
            # 清空梯度，set_to_none=True 比默认的zero_grad()更高效，因为它直接将梯度设为None而不是 0 张量
            optimizer.zero_grad(set_to_none=True)
        # 每隔log_interval步或在最后一步时记录日志
        if step % args.log_interval == 0 or step == iters - 1:
            # 还原真实的Loss数值用于显示 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            avg_reward = rewards.mean().item()
            avg_res_len = completion_mask.sum(dim=1).float().mean().item()
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"loss: {current_loss:.4f}, aux_loss: {current_aux_loss:.4f}, avg_reward: {avg_reward:.4f}, "
                   f"avg_response_len: {avg_res_len:.2f}, learning_rate: {current_lr:.8f}")
            if wandb and is_main_process():
                wandb.log({
                    "loss": current_loss,
                    "aux_loss": current_aux_loss,
                    "avg_reward": avg_reward,
                    "avg_response_len": avg_res_len,
                    "avg_advantages": advantages.mean().item(),
                    "learning_rate": current_lr
                })
        # 仅在主进程 (main process) 中保存模型，防止多卡训练时多个进程同时写文件导致冲突
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式 (影响Dropout, BatchNorm等)
            model.eval()
            lm_checkpoint(model.config, model=model, prefix="grpo", optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=step, wandb=wandb)
            # 切回训练模式
            model.train()
        # 显式删除当前步的变量，防止引用计数导致显存无法释放
        # 这在显存紧张的LLM训练中非常常见
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
    parser.add_argument("--use_compile", action="store_true", default=False, help="是否编译模型（默认不编译）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式环境（如果是多卡训练），获取当前进程的本地排名（local_rank，例如第0张卡还是第1张卡）
    local_rank = init_distributed_mode()
    # 如果分布式环境已初始化，指定当前进程使用的GPU设备
    if torch.distributed.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，确保可复现性。
    # 关键点：每张卡使用不同的种子（42 + rank），防止所有卡生成完全一样的数据增强或Dropout
    setup_seed(42 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))
    
    # ========== 2. 定义策略模型、参考模型和奖励模型==========
    # 实例化模型配置类，决定模型大小（隐藏层大小、层数）和是否使用混合专家（MoE）
    model = MiniGPTForCausalLM(MiniGPTConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=args.use_moe
    ))
    # 打印模型参数量
    get_model_params(model, model.config)
    # 优先从checkpoint加载断点数据，以便恢复训练
    ckp_data = lm_checkpoint(model.config, prefix="grpo", device=args.device) if args.from_resume else None
    # 加载基础模型
    if not ckp_data:
        model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"], strict=False)
        Logger(f"Load model weights from {args.from_weight}")
    # 定义参考模型，通常是SFT后的原始模型，避免策略模型偏离太远，遗忘SFT的内容
    ref_model = MiniGPTForCausalLM(model.config)
    ref_model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"])
    # 冻结模型参数，不参与权重更新
    ref_model = ref_model.eval().requires_grad_(False)
    # 定义奖励模型，对策略模型输出的一组回答打分
    reward_model = AutoModel.from_pretrained(args.reward_model_path, dtype=torch.float16, trust_remote_code=True)
    # 冻结模型参数，不参与权重更新
    reward_model = reward_model.eval().requires_grad_(False)
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 选择半精度类型：优先使用bfloat16（训练更稳定，范围更广），否则使用float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 创建自动混合精度（AMP）上下文管理器
    # 这将在后续的forward过程中自动将部分运算转为半精度以加速，并节省显存
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 定义数据和优化器 ==========
    tokenizer = AutoTokenizer.from_pretrained("../model")
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，必须使用DistributedSampler来确保每张卡分到不同的数据
    train_sampler = DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    # 统计优化器需要更新的总步数
    total_optimizer_steps = (len(loader_for_count) // args.accumulation_steps) * args.epochs
    # 余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume and ckp_data:
        model.load_state_dict(ckp_data["model"])
        scheduler.load_state_dict(ckp_data["scheduler"]) 
        optimizer.load_state_dict(ckp_data["optimizer"])
        # optimizer默认将数据加载到CPU中，需要手动切换到GPU中，避免梯度反向传播时，数据设备不一致导致报错
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
        # 获取断点前的epoch和step，以便恢复训练
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
    model.to(args.device)
    # [PyTorch 2.0特性] 编译模型，优化计算图，显著提升训练速度
    if args.use_compile:
        model = torch.compile(model)
        Logger("Enable torch.compile to speedup training")
    # 参考模型和奖励模型加载到指定计算设备上
    ref_model.to(args.device)
    reward_model.to(args.device)

    # ========== 6. 配置wandb ==========
    wandb = None
    # 仅在主进程（Master Node，通常是rank 0）中初始化WandB，避免多卡重复上传日志
    if args.use_wandb and is_main_process():
        # 这里使用了swanlab作为wandb的替代或别名
        import swanlab as wandb
        # 尝试从Checkpoint中恢复之前的run_id，这样图表能对接上
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        # 定义本次运行的名称，包含关键超参数
        wandb_run_name = f"grpo_epoch_{args.epochs}_bs_{args.batch_size}_lr_{args.learning_rate}"
        # 初始化项目
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 7. DDP包模型 ==========
    if torch.distributed.is_initialized():
        # 忽略旋转位置编码（RoPE）的预计算缓存，因为它们是常量，不需要同步梯度
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用DDP包装模型，负责多卡间的梯度同步
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP关键步骤：每个epoch设置不同的随机种子，确保数据shuffle顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 再次设置Python随机种子（为了后续indices生成），并生成随机索引
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 计算需要跳过的步数：只有在“恢复训练的那个epoch”才需要跳过之前的step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 使用自定义的SkipBatchSampler，实现精确到step的断点续训
        # 它会快速空转跳过前skip个batch，直接从断点处开始产出数据
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step, 从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # 训练结束，销毁进程组，释放资源
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()