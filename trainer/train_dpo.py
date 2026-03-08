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
    """将模型的原始输出logits转换为对应label的log概率。
    这对于计算DPO Loss至关重要，因为我们需要知道模型生成特定token的概率。
    logits shape: (batch_size, seq_len, vocab_size)
    labels shape: (batch_size, seq_len)
    
    Q：为什么要使用log概率？
    A：是为了方便从“单个词的概率”到“整句话的概率”的转换。
    在概率论中，计算一个序列（一句话）的生成概率，是将其中每个词（Token）的概率相乘。
    但是因为是在对数空间 (Log Space) 操作，所以乘法就变成了加法。
    
    具体步骤如下：
    假设一句话只有三个词 "I", "love", "AI"。
    模型生成这句话的概率 P(sentence) 是每个词概率的乘积：
            P(sentence)=P("I") * P("love") * P("AI")
    如果我们对两边取对数（Log），乘号就变成了加号：
            logP(sentence)=logP("I") + logP("love") + logP("AI")
    """
    # 对logits进行log_softmax归一化，得到所有词汇的log概率
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    # gather操作：只提取labels对应位置的那个token的概率
    # .unsqueeze(2) 将labels维度变为 (batch_size, seq_len, 1) 以匹配gather需求
    # .squeeze(-1) 将结果变回 (batch_size, seq_len)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    # log_probs_per_token shape: (batch_size, seq_len)
    return log_probs_per_token

def get_dpo_loss(ref_log_probs, actor_log_probs, mask, beta):
    """DPO Loss的核心计算函数
    ref_log_probs和actor_log_probs形状均为: (batch_size, seq_len)
    这里的batch_size实际上包含了chosen和rejected拼接后的数量 (即 2 * 真实batch_size)
    """
    # 1. 计算每个样本的有效长度（排除padding）
    # clamp_min(1e-8) 是为了防止全 padding 的异常数据，导致除以 0
    seq_length = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # 2. 计算整个句子的平均log概率
    # 句子中所有有效Token的log概率加起来，就得到整句话的总log概率
    # 乘以mask是为了把padding部分的概率置为 0，然后求和并除以有效长度
    # Q: 为什么要除以seq_length.squeeze()？
    # A: 首先log概率通常为负数，越接近0越好；其次长句子天然比短句子的sum值更小，
    # 如果不做归一化，DPO可能会错误的认为短句子总是比长句子更好。除以句子的有效
    # 长度后，计算的是平均每个token的log概率，这样长短句就可以公平比较了。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_length.squeeze()
    actor_log_probs = (actor_log_probs * mask).sum(dim=1) / seq_length.squeeze()

    # 3. 将拼接在一起的数据拆分回 chosen (好回答) 和 rejected (坏回答)
    # 假设输入batch是 [chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_actor_log_probs = actor_log_probs[:batch_size // 2]
    reject_actor_log_probs = actor_log_probs[batch_size // 2:]

    # 4. 计算策略模型对 好回答 和 坏回答 的log概率差
    pi_logratios = chosen_actor_log_probs - reject_actor_log_probs

    # 5. 计算参考模型对 好回答 和 坏回答 的log概率差 (这是基准)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # 6. 计算DPO的核心logits
    # 含义：策略模型相对于参考模型，是否更偏向于“好回答”
    logits = pi_logratios - ref_logratios

    # 7. 计算损失：使用log sigmoid
    # beta是超参数，控制由于偏离参考模型带来的惩罚力度
    loss = -1 * torch.nn.functional.logsigmoid(beta * logits)
    return loss.mean()

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 记录Epoch开始时间，用于计算预计剩余时间 (ETA)
    start_time = time.time()
    # start=start_step + 1 确保显示的step计数是从断点后续开始的
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        # 将chosen和rejected拼在一起，一次性送入模型计算，提高效率
        # shape变为 (batch_size * 2, seq_len)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        # 根据当前的总步数 (epoch * iters + step)，计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 手动更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 进入自动混合精度上下文（通常是torch.cuda.amp.autocast）
        with autocast_ctx:
            # 参考模型只前向计算，不参与反向梯度传播
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 计算参考模型的log概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            # 策略模型的前向计算
            actor_outputs = model(x)
            actor_logits = actor_outputs.logits
            # 计算策略模型的log概率
            actor_log_probs = logits_to_log_probs(actor_logits, y)
            dpo_loss = get_dpo_loss(ref_log_probs, actor_log_probs, mask, beta=args.beta)
            # 总损失 = DPO损失 + 辅助损失 (aux_loss通常用于MoE的负载均衡，Dense模型时为 0)
            loss = dpo_loss + actor_outputs.aux_loss
            # 将loss除以累计步数，这样累积后的梯度和才是预期的均值
            loss = loss / args.accumulation_steps
        # scaler用于处理FP16下的梯度下溢问题 (Gradient Underflow)
        scaler.scale(loss).backward()
        # 梯度更新 (Optimizer Step) - 仅在达到累积步数时执行
        if (step + 1) % args.accumulation_steps == 0:
            # 先将梯度unscale(反缩放) 回FP32，以便进行梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪 (Gradient Clipping)，防止梯度爆炸，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新模型参数
            scaler.step(optimizer)
            # 更新scaler的缩放因子 (scale factor)
            scaler.update()
            # 清空梯度，set_to_none=True 比默认的zero_grad()更高效，因为它直接将梯度设为None而不是 0 张量
            optimizer.zero_grad(set_to_none=True)
        # 每隔log_interval步或在最后一步时记录日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实的Loss数值用于显示 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss.item()
            current_aux_loss = actor_outputs.aux_loss.item() if actor_outputs.aux_loss is not None else 0.0
            current_lr = optimizer.param_groups[-1]["lr"]
            # 计算ETA (预计剩余时间)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, "
                   f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta_time: {eta_min:.1f} min")
            if wandb: 
                wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, 
                    "learning_rate": current_lr, "eta_time (minutes)": eta_min})
        # 仅在主进程 (main process) 中保存模型，防止多卡训练时多个进程同时写文件导致冲突
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式 (影响Dropout, BatchNorm等)
            model.eval()
            lm_checkpoint(model.config, model=model, prefix="dpo", optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb)
            # 切回训练模式
            model.train()
        # 显式删除当前步的变量，防止引用计数导致显存无法释放
        # 这在显存紧张的LLM训练中非常常见
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
    
    # ========== 2. 定义模型、参考模型 ==========
    # 实例化模型配置类，决定模型大小（隐藏层大小、层数）和是否使用混合专家（MoE）
    model = MiniGPTForCausalLM(MiniGPTConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=args.use_moe
    ))
    # 打印模型参数量
    get_model_params(model, model.config)
    # 优先从checkpoint加载断点数据，以便恢复训练
    ckp_data = lm_checkpoint(model.config, prefix="dpo", device=args.device) if args.from_resume else None
    # 加载基础模型
    if not ckp_data:
        model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"], strict=False)
        Logger(f"Load model weights from {args.from_weight}")
    # 定义参考模型，通常是SFT后的原始模型
    ref_model = MiniGPTForCausalLM(model.config)
    ref_model.load_state_dict(torch.load(args.from_weight, map_location=args.device)["model"])
    # 冻结模型参数，不参与权重更新
    ref_model.eval().requires_grad_(False)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 选择半精度类型：优先使用bfloat16（训练更稳定，范围更广），否则使用float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 创建自动混合精度（AMP）上下文管理器
    # 这将在后续的forward过程中自动将部分运算转为半精度以加速，并节省显存
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 定义数据和优化器 ==========
    tokenizer = AutoTokenizer.from_pretrained("../model")
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，必须使用DistributedSampler来确保每张卡分到不同的数据
    train_sampler = DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
    # 定义梯度缩放器（Scaler），用于float16训练，防止梯度下溢
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume and ckp_data:
        model.load_state_dict(ckp_data["model"])
        scaler.load_state_dict(ckp_data["scaler"])
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
    ref_model.to(args.device)
    
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
        wandb_run_name = f"dpo_epoch_{args.epochs}_bs_{args.batch_size}_lr_{args.learning_rate}"
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