import math
import torch
from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

class MiniGPTConfig(PretrainedConfig):
    model_type = "minigpt"  # 模型类型标识，用于Hugging Face库识别模型类别，便于自动加载对应的模型结构
    def __init__(
            self,
            dropout: float = 0.0,  # 在现代大模型预训练中，为了最大化数据拟合能力，通常将丢弃率设为0
            bos_token_id: int = 1,  # 起始Token ID，代表"Begin of Sentence"，每句话的开头
            eos_token_id: int = 2,  # 结束Token ID，代表"End of Sentence"，告诉模型生成结束
            hidden_act: str = "silu",  # 激活函数，采用SiLU (Swish) ，效果优于ReLU
            hidden_size: int = 512,  # 隐藏层维度，每个Token向量的宽度，决定了模型的“宽度”
            intermediate_size: int = None,  # FFN 中间层维度，通常设为hidden_size的4倍；如果为None，则模型初始化时自动计算
            max_position_embeddings: int = 32768,  # 最大上下文长度32K（已包含YaRN算法的长度外推）
            num_attention_heads: int = 8,  # 查询(Query)头数
            num_hidden_layers: int = 8,  # 网络层数，Transformer Block堆叠的数量，代表模型的“深度”
            num_key_value_heads: int = 2,  # 键值(KV)头数，当此值小于num_attention_heads时，即开启了GQA
            vocab_size: int = 6400,  # 词表大小
            rms_norm_eps: float = 1e-05,  # 归一化稳定性系数，防止分母为0，保证数值计算的稳定性
            rope_theta: int = 1000000.0,  # 位置编码的基频
            inference_rope_scaling: bool = False,  # 推理时启用YaRN算法进行长度外推
            flash_attn: bool = True,  # 使用Flash Attention加速算子，能大幅提升训练和推理速度，并节省显存
            use_moe: bool = False,  # 是否启用MoE混合专家模型
            num_experts_per_tok: int = 2,  # 每个Token在推理时实际会激活的路由专家数量
            n_routed_experts: int = 4,  # 可供选择的路由专家总数量
            n_shared_experts: int = 1,  # 共享专家数量
            scoring_func: str = "softmax",  # Router网络用来计算每个专家权重概率的函数
            aux_loss_alpha: float = 0.01,  # 辅助损失系数，防止Router总是只选某几个专家
            seq_aux: bool = True,  # 计算辅助损失的范围是在整个序列级别上统计，而非仅针对单个Token
            norm_topk_prob: bool = True,  # 选出Top-K个专家后，是否将这K个专家的权重重新归一化
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        # 以下参数适用于MOE模型
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

class RMSNorm(torch.nn.Module):
    """均方根归一化
    RMSNorm是LayerNorm的简化版本，只对输入进行缩放，不进行中心化（不减去均值）。
    相比LayerNorm，RMSNorm计算更高效，且在实际应用中效果相当。
    公式：
        RMSNorm(x) = weight * (x / sqrt(mean(x^2) + eps))
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化RMSNorm层
        Args:
            dim: 输入特征的维度
            eps: 防止除零的小常数（默认1e-5）
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全 1
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """计算RMS归一化
        公式：
            x / sqrt(mean(x^2) + eps)  
        Args:
            x: 输入张量 [..., dim]

        Returns:
            归一化后的张量，形状与输入相同
        """
        # torch.rsqrt: 计算 1/sqrt，比先sqrt再除更快
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入张量，可以是任意精度（float16, bfloat16, float32）
            
        Returns:
            归一化并缩放后的张量，保持原始精度
        """
        # 先转换为float32进行归一化计算（提高数值稳定性），然后转换回原始精度（type_as(x)）
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """预计算RoPE (Rotary Position Embedding)的频率矩阵
    RoPE通过旋转矩阵将位置信息编码到Query和Key中，使模型能够理解token的相对位置。
    本函数预计算所有位置的cos和sin值，避免在每次前向传播时重复计算。
    支持YaRN (Yet another RoPE extensioN)外推方法，可以处理超过训练时最大长度的序列。
    
    Args:
        dim: 每个注意力头的维度（head_dim）
        end: 最大序列长度（默认 32768）
        rope_base: RoPE的基频率参数（默认 1e6）
        rope_scaling: RoPE外推配置字典（YaRN方法），如果为None则不使用外推
        
    Returns:
        freqs_cos: 预计算的cos值 [end, dim]
        freqs_sin: 预计算的sin值 [end, dim]
    """
    # ========== 步骤 1：计算基础频率 ==========
    # RoPE频率公式：f_i = 1 / (rope_base^(2i/dim))
    # 其中i是维度索引（0, 2, 4, ..., dim-2），只使用偶数索引
    # 频率随维度索引增加而递减，形成不同频率的旋转
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    
    # ========== 步骤 2：应用YaRN外推（如果启用） ==========
    if rope_scaling is not None:
        # 获取YaRN配置参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 如果目标长度超过训练长度，应用YaRN外推
        if end / orig_max > 1.0:
            # YaRN公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 其中γ是线性斜坡函数，s是缩放因子（factor）
            # 对于低频维度（i < low），不进行缩放
            # 对于高频维度（i > high），完全缩放
            # 对于中间维度，线性插值
            
            # 计算频率调整的边界维度，inv_dim(b) 返回频率为b的维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 计算线性斜坡函数γ，对于维度i：γ(i) = (i - low) / (high - low)，限制在[0, 1]
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 应用YaRN缩放：f'(i) = f(i) * ((1-γ) + γ/s)
            freqs = freqs * (1 - ramp + ramp / factor)

    # ========== 步骤 3：计算所有位置的频率 ==========
    # 为每个位置计算频率：freqs[pos, dim] = pos * freqs[dim]
    t = torch.arange(end, device=freqs.device)
    # 外积：[end, dim//2]
    freqs = torch.outer(t, freqs).float()

    # ========== 步骤 4：计算cos和sin值 ==========
    # 将频率转换为cos和sin值，用于旋转矩阵
    # 由于RoPE使用复数旋转，需要将dim//2的频率复制到完整的dim维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码（RoPE）到Query和Key
    RoPE通过复数旋转将位置信息编码到Q和K中：
    R_θ(x) = [x_0*cos(θ) - x_1*sin(θ), x_0*sin(θ) + x_1*cos(θ)]
    在实现中，将复数旋转分解为实部和虚部的线性组合，使用rotate_half函数实现。
    
    Args:
        q: Query张量 [batch, seq_len, num_heads, head_dim]
        k: Key张量 [batch, seq_len, num_kv_heads, head_dim]
        cos: 预计算的cos值 [seq_len, head_dim]
        sin: 预计算的sin值 [seq_len, head_dim]
        position_ids: 位置索引（未使用，cos/sin已包含位置信息）
        unsqueeze_dim: 在哪个维度插入新维度以匹配q/k的形状（默认 1）
        
    Returns:
        q_embed: 应用RoPE后的Query [batch, seq_len, num_heads, head_dim]
        k_embed: 应用RoPE后的Key [batch, seq_len, num_kv_heads, head_dim]
    """
    def rotate_half(x):
        """旋转向量的后半部分
        将向量分成两半，交换位置并取反后半部分：
        [a, b, c, d] -> [-c, -d, a, b]
        这实现了复数旋转的实部/虚部交换。
        
        Args:
            x: 输入张量，最后一个维度会被分成两半
            
        Returns:
            旋转后的张量，形状与输入相同
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用RoPE旋转
    # 公式：R_θ(x) = x * cos(θ) + rotate_half(x) * sin(θ)
    # 这等价于复数旋转：x * e^(iθ) = x * (cos(θ) + i*sin(θ))
    # 其中rotate_half实现了虚部的操作
    # unsqueeze用于调整cos和sin的形状以匹配q/k：[seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复Key/Value heads以实现GQA
    GQA是一种注意力机制优化，使用较少的KV heads来匹配更多的Query heads。
    例如：8个Query heads对应2个 KV heads，每个KV head需要重复4次。
    这样可以减少KV缓存的大小，在推理时节省显存。
    
    Args:
        x: Key或Value张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 每个KV head需要重复的次数（n_rep = num_heads / num_kv_heads）
        
    Returns:
        重复后的张量 [batch, seq_len, num_heads, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # [B, L, num_kv_heads, 1, D]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # [B, L, num_kv_heads, n_rep, D]
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # [B, L, num_heads, D]
    )

class Attention(torch.nn.Module):
    """多头注意力机制（支持GQA和Flash Attention）
    实现了标准的缩放点积注意力（Scaled Dot-Product Attention），支持：
    1. GQA (Grouped Query Attention): 使用较少的KV heads匹配更多的Query heads
    2. Flash Attention: 使用PyTorch 2.0+的优化注意力实现
    3. RoPE: 通过旋转位置编码将位置信息注入Q和K
    4. KV Cache: 支持推理时的KV缓存加速
    
    注意力公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, args: MiniGPTConfig):
        super().__init__()
        # ========== GQA 配置 ==========
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 确保Query heads数量能被KV heads数量整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # Query heads数量
        self.n_local_heads = args.num_attention_heads
        # KV heads数量
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个KV head需要重复的次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # ========== 投影层 ==========
        # Q 投影：hidden_size -> num_heads * head_dim        
        self.q_proj = torch.nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # K 投影：hidden_size -> num_kv_heads * head_dim（GQA：较少的 heads）
        self.k_proj = torch.nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # V 投影：hidden_size -> num_kv_heads * head_dim（GQA：较少的 heads）
        self.v_proj = torch.nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：num_heads * head_dim -> hidden_size
        self.o_proj = torch.nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # ========== Dropout ==========
        self.attn_dropout = torch.nn.Dropout(args.dropout)
        self.resid_dropout = torch.nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # ========== Flash Attention ==========
        # 检查是否支持 Flash Attention（需要 PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """前向传播
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            position_embeddings: RoPE位置编码 (cos, sin)元组
            past_key_value: 缓存的KV值，用于增量解码 [batch, past_len, num_kv_heads, head_dim]
            use_cache: 是否返回KV缓存供下次使用
            attention_mask: 注意力掩码 [batch, seq_len]，1 表示有效位置，0 表示掩码位置
            
        Returns:
            output: 注意力输出 [batch, seq_len, hidden_size]
            past_kv: 新的KV缓存（如果use_cache=True），否则为 None
        """
        bsz, seq_len, _ = x.shape

        # ========== 步骤 1：Q/K/V 投影 ==========
        # 将输入投影到 Q、K、V 空间
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑为多头格式：[batch, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ========== 步骤 2：应用RoPE位置编码 ==========
        # 将位置信息编码到Q和K中
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ========== 步骤 3：KV Cache处理 ==========
        # 如果有缓存的KV 值（增量解码），将其与当前KV拼接
        if past_key_value is not None:
            # past_key_value[0]是缓存的K，past_key_value[1]是缓存的V
            # 在序列维度（dim=1）上拼接：[batch, past_len+seq_len, num_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果需要缓存，保存当前的KV值
        past_kv = (xk, xv) if use_cache else None

        # ========== 步骤 4：GQA处理 ==========
        # 调整维度顺序为 [batch, num_heads, seq_len, head_dim]（Flash Attention 格式）
        # 对于KV，需要重复heads以匹配Query heads数量
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # ========== 步骤 5：计算注意力 ==========
        # 优先使用Flash Attention（如果支持且条件满足）
        # 条件1：序列长度 > 1
        # 条件2：没有KV cache
        # 条件3：没有复杂掩码（因为scaled_dot_product_attention会自动使用因果掩码）
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 标准注意力计算流程
            # 步骤 5.1：计算注意力分数 QK^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 步骤 5.2：应用因果掩码（只对当前序列部分）
            # 上三角矩阵掩码，防止看到未来的token
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 步骤 5.3：应用注意力掩码（如果有）
            if attention_mask is not None:
                # 将掩码扩展到 [batch, 1, 1, seq_len]，并转换为分数掩码
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 0 -> -inf, 1 -> 0
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            # 步骤 5.4：Softmax归一化
            scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)

            # 步骤 5.5：应用dropout            
            scores = self.attn_dropout(scores)

            # 步骤 5.6：加权求和
            # [batch, num_heads, seq_len, head_dim]
            output = scores @ xv

        # ========== 步骤 6：输出投影 ==========
        # 重塑并投影回hidden_size
        # [batch, seq_len, num_heads * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # [batch, seq_len, hidden_size]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(torch.nn.Module):
    """SwiGLU 前馈网络
    实现了SwiGLU (Swish-Gated Linear Unit)激活函数的前馈网络。
    SwiGLU是GLU (Gated Linear Unit)的变体，使用Swish/SiLU作为门控激活函数。
    公式：
        FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))
    其中：
        - gate_proj: 门控投影，用于生成门控信号
        - up_proj: 上投影，用于生成特征
        - Swish(x) = x * sigmoid(x) = x * silu(x)
        - down_proj: 下投影，将中间维度映射回hidden_size

    相比标准FFN (ReLU(xW1)W2)，SwiGLU通常有更好的性能。
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        # ========== 中间层维度计算 ==========
        # 如果未指定intermediate_size，则自动计算        
        if config.intermediate_size is None:
            # 标准比例：intermediate_size = hidden_size * 8/3
            # 例如：hidden_size = 512 -> intermediate_size ≈ 1365
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到64的倍数（优化GPU计算效率）
            # 例如：1365 -> 1408 (64 * 22)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # ========== 投影层 ==========
        # 门控投影，hidden_size -> intermediate_size
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 下投影，intermediate_size -> hidden_size
        self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 上投影，hidden_size -> intermediate_size
        self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        # ========== Dropout 和激活函数 ==========
        self.dropout = torch.nn.Dropout(config.dropout)
        # 激活函数：通常是 "silu" (Swish)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """前向传播
        SwiGLU公式：FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MoEGate(torch.nn.Module):
    """MoE (Mixture of Experts)门控网络
    负责为每个token选择top-k个专家，并计算专家权重。
    使用辅助损失（auxiliary loss）来鼓励专家负载均衡，防止专家退化。
    
    工作流程：
        1. 计算每个专家对每个token的分数（logits）
        2. 使用softmax转换为概率
        3. 选择top-k个专家
        4. 计算辅助损失（训练时）
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        # 每个token选择的专家数量
        self.top_k = config.num_experts_per_tok
        # 专家总数
        self.n_routed_experts = config.n_routed_experts
        # 评分函数
        self.scoring_func = config.scoring_func
        # 辅助损失权重
        self.alpha = config.aux_loss_alpha
        # 是否在序列级别计算辅助损失
        self.seq_aux = config.seq_aux
        # 是否标准化top-k概率
        self.norm_topk_prob = config.norm_topk_prob
        # 门控网络输入维度
        self.gating_dim = config.hidden_size
        # 门控网络权重：[n_routed_experts, hidden_size]，每一行对应一个专家的权重向量
        self.weight = torch.nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用Kaiming均匀分布初始化权重"""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """前向传播
        Args:
            hidden_states: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            topk_idx: 选择的专家索引 [batch*seq_len, top_k]
            topk_weight: 专家权重 [batch*seq_len, top_k]
            aux_loss: 辅助损失（标量），用于鼓励负载均衡
        """
        bsz, seq_len, h = hidden_states.shape

        # ========== 步骤 1：计算专家分数 ==========
        
        # view(-1, h): 改变张量形状为 [batch * seq_len, h]。
        # 把所有句子的所有词平铺开，变成一个长长的列表，因为我们对每个词是独立处理的。
        hidden_states = hidden_states.view(-1, h)
        # 计算每个Token和每个Expert的匹配分数（原始分数，未归一化）
        logits = torch.nn.functional.linear(hidden_states, self.weight, None)

        # ========== 步骤 2：转换为概率 ==========
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported scoring function for MoE gating: {self.scoring_func}")

        # ========== 步骤 3：选择top-k专家 ==========
        # topk_weight: [batch*seq_len, top_k] 选中的那k个专家的概率值
        # topk_idx: [batch*seq_len, top_k] 选中的那k个专家的索引（ID 号）
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # ========== 步骤 4：标准化top-k概率（可选） ==========
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ========== 步骤 5：计算辅助损失（训练时） ==========
        # 辅助损失用于鼓励专家负载均衡，防止某些专家被过度使用或完全不用
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # === 方案 A：序列级辅助损失 (DeepSeek-V2/V3 常用) ===
                # 这种计算方式更精细，在每条样本内部看负载均衡
                # 变形回 [batch, seq_len, n_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 计算每个专家的使用频率（期望负载）
                # 创建一个全 0 矩阵用来统计次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_: 这是一个复杂的“散射加法”操作，形象的理解，这是在“投票”。
                # topk_idx_for_aux_loss里的值是专家ID，它告诉我们每个Token投给了谁。
                # 这行代码统计：在这个Batch里，每个专家被选中了多少次。
                # .div_(...): 除以期望的平均次数，将其归一化。
                # 如果 ce = 1，说明该专家被选中的频率正好等于平均水平。
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                    ).div_(seq_len * aux_topk / self.n_routed_experts)
                # 计算损失：(实际使用频率 * 专家平均概率得分)
                # 这种损失设计会迫使模型倾向于让所有专家的使用频率和平均得分趋于一致。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # === 方案 B：Token级辅助损失 (传统的Switch Transformer做法) ===
                # 这种是全局统计所有Token。
                # one_hot 独热编码，如果ID是3，变成 [0, 0, 0, 1, 0...]
                mask_ce = torch.nn.functional.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 每个专家的平均使用频率 [n_routed_experts]
                ce = mask_ce.float().mean(0)
                # 每个专家的平均分数 [n_routed_experts]，模型“想”选它的程度
                Pi = scores_for_aux.mean(0)
                # 计算负载均衡分数
                fi = ce * self.n_routed_experts
                # 经典的负载均衡损失公式：
                # minimize (N * sum(Pi * fi))
                # 只有当概率分布是均匀分布时，这个点积最小。
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 如果不在训练，或者不需要辅助损失，损失为 0
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(torch.nn.Module):
    """MoE (Mixture of Experts)前馈网络
    使用多个专家（FeedForward）处理不同的token，通过门控网络动态选择专家。
    支持路由专家（routed experts）和共享专家（shared experts）两种类型。
    
    工作流程：
        1. 门控网络为每个token选择top-k个路由专家
        2. 每个token被路由到选中的专家处理
        3. 专家输出按权重加权求和
        4. 共享专家处理所有token并添加到输出
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # ========== 路由专家 ==========
        # 通过门控网络动态选择，每个token只使用top-k个专家
        self.experts = torch.nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])

        # ========== 门控网络 ==========
        # 负责为每个token选择专家并计算权重
        self.gate = MoEGate(config)

        # ========== 共享专家 ==========
        # 处理所有token，不经过门控网络
        # 用于提供通用特征，增强模型表达能力
        if config.n_shared_experts > 0:
            self.shared_experts = torch.nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """前向传播
        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            
        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        # 保存原始输入，用于共享专家
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # ========== 步骤 1：门控网络选择专家 ==========
        # 专家索引 topk_idx: [batch*seq_len, top_k]
        # 专家权重 topk_weight: [batch*seq_len, top_k]
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # ========== 步骤 2：路由到专家处理 ==========
        # [batch*seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # 展平的专家索引 [batch*seq_len*top_k]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式：为每个token的每个选中专家复制输入
            # 例如：top_k=2，每个token需要处理2次
            # [batch*seq_len*top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            # 对每个专家，处理分配给它的token
            for i, expert in enumerate(self.experts):
                # 找到分配给专家i 的token索引
                mask = flat_topk_idx == i
                expert_out = expert(x[mask])
                if expert_out.shape[0] > 0:
                    # 如果有token分配给该专家，保存输出
                    y[mask] = expert_out.to(y.dtype)
                else: 
                    # 如果没有token分配给该专家，创建空输出（保持梯度流）
                    y[mask] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            # 每个token的top-k个专家输出加权平均
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # [batch*seq_len, hidden_size]
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用优化的推理函数
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # ========== 步骤 3：添加共享专家输出 ==========
        # 共享专家处理所有token，输出直接添加到结果中
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                # 残差连接
                y = y + expert(identity)
        # 保存辅助损失供后续使用
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """优化的MoE推理函数（仅推理时使用）
        通过批量处理每个专家的所有token，减少计算开销。
        工作流程：
            1. 按专家索引排序token
            2. 统计每个专家处理的token数量
            3. 批量处理每个专家的所有token
            4. 按权重加权并累加到输出缓存
        
        Args:
            x: 输入张量 [batch*seq_len, hidden_size]
            flat_expert_indices: 展平的专家索引 [batch*seq_len*top_k]
            flat_expert_weights: 展平的专家权重 [batch*seq_len*top_k, 1]
            
        Returns:
            输出张量 [batch*seq_len, hidden_size]
        """
        # 输出缓存
        expert_cache = torch.zeros_like(x)

        # ========== 步骤 1：按专家索引排序 ==========
        # 将token按专家索引排序，使同一专家的token聚集在一起
        idxs = flat_expert_indices.argsort()

        # ========== 步骤 2：统计每个专家处理的token数量 ==========
        # bincount: 统计每个专家被选中的次数
        # cumsum: 累积和，得到每个专家的token范围
        # 例如：[6, 15, 20, 26] 表示：
        #   - 专家0 处理前6个token
        #   - 专家1 处理第6-15个token
        #   - 专家2 处理第15-20个token
        #   - 专家3 处理第20-26个token
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token的原始索引（去除top_k的重复）
        token_idxs = idxs // self.config.num_experts_per_tok

        # ========== 步骤 3：批量处理每个专家 ==========
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果该专家没有处理的token，则跳过
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            # 原始token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 该专家需要处理的token
            expert_tokens = x[exp_token_idx]
            # 批量处理该专家的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 应用权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到输出缓存（使用scatter_add处理同一token被多个专家处理的情况）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache

class MiniGPTBlock(torch.nn.Module):
    """Transformer的标准层
    结构采用Pre-Norm设计：
    Input -> Norm -> Attention -> Residual Add
    Input -> Norm -> FFN/MoE -> Residual Add
    """
    def __init__(self, layer_id: int, config: MiniGPTConfig):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 自注意力层
        self.self_attn = Attention(config)
        # MLP层（FFN或MOE）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config) 
        # Attention层的归一化，采用Pre-Norm设计
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP层的归一化，采用Pre-Norm设计
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniGPTModel(torch.nn.Module):
    """MiniGPT 主模型核心类
    这是Transformer的Decoder-Only架构实现（类似 LLaMA 结构）。
    它负责将输入的Token IDs转换为深层的语义特征表示（Hidden States）。

    维度符号约定：
        B (Batch Size): 批次大小 (推理时通常为 1)
        S (Seq Length): 当前输入的序列长度 (训练时为全长，推理Decoding阶段为 1)
        H (Hidden Size): 隐藏层维度 (如 512)
        V (Vocab Size): 词表大小 (如 6400)
        L (Layers): 层数 (如 8)
        HD (Head Dim): 单个注意力头的维度 (H // num_heads)
        MaxPos: 最大支持序列长度

    主要流程：
    Input IDs -> Embeddings -> [Transformer Blocks x L] -> RMSNorm -> Output Hidden States
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # ========== 1. 词嵌入层 (Embedding) ==========
        # 将离散的Token ID映射为稠密向量，形状 [V, H]
        # 注意：在MiniGPTForCausalLM中，这个权重通常与输出层的lm_head共享 (Weight Tying)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        # Dropout层，用于防止过拟合
        self.dropout = torch.nn.Dropout(config.dropout)

        # ========== 2. 堆叠Transformer层 (Layers) ==========
        # 使用ModuleList存储 L 个MiniMindBlock
        # 每个Block包含 Attention 和 FFN/MoE
        self.layers = torch.nn.ModuleList([MiniGPTBlock(l, config) for l in range(self.num_hidden_layers)])

        # ========== 3. 最终归一化层 (Final Norm) ==========
        # 在输出之前进行最后一次RMSNorm，这是LLaMA架构的标准做法
        # 形状 [H]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ========== 4. 预计算RoPE位置编码 (Precompute RoPE) ==========
        # 预先计算所有可能位置的cos 和 sin值，避免前向传播时重复计算
        # freqs_cos/sin 形状: [MaxPos, HD]
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling)
        # 将频率表注册为buffer
        # buffer不会被视为模型参数(parameter)，不参与梯度更新，但会随模型权重文件保存
        # persistent=False 表示这些值可以根据config动态重新计算，不强制依赖权重文件
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """前向传播
        Args:
            input_ids: 输入序列 [B, S]，训练时S是整个句子长度，推理Decoding阶段S通常为 1。
            attention_mask: 掩码 [B, S]。
            past_key_values: 历史KV缓存列表。
                            List 长度为 L，每个元素是 (K, V) 元组。
                            K/V 形状: [B, Past_Len, Num_KV_Heads, HD]。
            use_cache: 是否开启KV Cache加速 (推理时为 True)。

        Returns:
            hidden_states: [B, S, H] 模型输出特征
            presents: 新的KV Cache列表
            aux_loss: MoE负载均衡辅助损失
        """
        batch_size, seq_length = input_ids.shape

        # ========== KV Cache 兼容性处理 ==========
        # 如果传入的是Hugging Face新版的高级Cache对象 (含有 .layers 属性)
        # MiniGPT暂时不支持，为了防止报错，强制清空缓存 (安全降级)
        if hasattr(past_key_values, "layers"): past_key_values = None
        # 初始化 past_key_values
        # 如果没有缓存 (Prefill阶段或训练阶段)，初始化为全 None 的列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # ========== 计算起始位置 (start_pos) ==========
        # 这里的逻辑是确定当前输入的Token在整篇文章中的绝对位置索引
        # 1. 如果有缓存 (past_key_values[0] 不为 None):
        #    说明是推理的 Decoding 阶段。
        #    past_key_values[0][0] 是第 0 层的 Key Tensor，形状 [B, Past_Len, H_kv, HD]
        #    .shape[1] 就是 Past_Len (历史已经处理过的 Token 数量)
        #    这也是当前新Token的起始索引。
        # 2. 如果没有缓存:
        #    说明是Prefill阶段或训练阶段，从第 0 个位置开始。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # ========== Token Embedding ==========
        # 将ID转换为向量: [B, S] -> [B, S, H]
        # 此时hidden_states包含了语义信息，但还没有位置信息
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ========== 提取位置编码 (RoPE Slicing) ==========
        # 根据绝对位置start_pos和当前长度seq_length，从预计算的表中切片
        # 切片范围: [start_pos : start_pos + seq_length]
        # 场景 A (训练/Prefill): start_pos=0, seq_len=N -> 取出前N个位置编码
        # 场景 B (推理 Decoding): start_pos=N, seq_len=1 -> 仅取出第N个位置的编码 (长度为 1)
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # ========== 逐层前向传播 ==========
        # 用于收集每一层新的KV Cache
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 进入Transformer Block
            # 输入: hidden_states [B, S, H]
            # 输出: 
            #   hidden_states: 更新后的特征 [B, S, H]
            #   present: 当前层更新后的KV Cache (包含历史+当前), 形状 [B, Past_Len+S, H_kv, HD]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # ========== 最终归一化 ==========
        # 经过所有层后，进行最后一次RMSNorm
        # [B, S, H] -> [B, S, H]
        hidden_states = self.norm(hidden_states)

        # ========== 汇总MoE辅助损失 ==========
        # 检查每一层，如果是MoE层 (MOEFeedForward)，提取其aux_loss
        # 将所有层的aux_loss相加，用于训练时的反向传播
        # 如果没有使用MoE，总aux_loss为 0
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniGPTForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniGPT 因果语言模型 (Causal Language Model)
    这是面向最终任务（文本生成）的顶层封装类。
    架构组成：
        Input IDs -> [MiniGPTModel] -> Hidden States -> [LM Head] -> Logits
    
    关键特性：
        1. 权重共享 (Weight Tying): 输入Embedding和输出LM Head共享同一份参数，显著减少显存。
        2. 推理优化 (Logits Slicing): 支持只计算最后一个Token的Logits，避免全量计算。
        3. 训练并行 (Parallel Training): 利用Mask实现一次性计算所有Token的Loss。
    """
    # 指定配置类，Hugging Face框架自动加载机制需要
    config_class = MiniGPTConfig

    def __init__(self, config: MiniGPTConfig = None):
        # 如果没有传入config，则实例化一个默认配置
        self.config = config or MiniGPTConfig()
        # 初始化父类 PreTrainedModel (负责权重加载、保存、下载等)
        super().__init__(self.config)

        # ========== 1. 骨干网络 (Backbone) ==========
        # 实例化纯Transformer Decoder，负责提取深层语义特征
        # 输入: [Batch, Seq_Len] -> 输出: [Batch, Seq_Len, Hidden_Size]
        self.model = MiniGPTModel(self.config)

        # ========== 2. 语言模型头 (LM Head) ==========
        # 这是一个线性投影层 (Linear Layer)
        # 作用: 将高维特征向量 (Hidden State) 映射回词表空间 (Vocab Space)
        # 形状: [Hidden_Size] -> [Vocab_Size]
        # bias=False: 现代大模型 (LLaMA等) 通常不使用偏置项，以提升数值稳定性
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # ========== 3. 权重共享 (Weight Tying) ==========
        # [重要优化] 将Input Embedding的权重指针指向LM Head的权重
        # 物理意义: 语义上，“输入一个词”和“预测一个词”使用的是同一个语义空间。
        # 显存优势: 词表通常很大 (如 64k)，权重共享能节省大量参数 (Hidden * Vocab)。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """前向传播 (支持训练和推理两种模式)
        Args:
            input_ids: 输入序列 [Batch, Seq_Len]。
                       - 训练时: 是一整句话 (Seq_Len = N)。
                       - 推理时(Decoding): 通常只是最新生成的那个词 (Seq_Len = 1)。
            attention_mask: 掩码 [Batch, Seq_Len] (1=有效, 0=padding)。
            labels: 标签序列 [Batch, Seq_Len]。
                    - 如果提供此参数，模型会计算Loss (训练模式)。
                    - 如果为None，只返回Logits (推理模式)。
            past_key_values: KV Cache列表。
                    - 用于存储历史Token的Key/Value，避免重复计算。
            use_cache: 是否返回更新后的KV Cache (推理时开启)。
            logits_to_keep: 【性能优化参数】
                    - 0 (默认): 计算所有Token的Logits (训练时必须选这个)。
                    - 1 (常用): 只计算最后一个Token的Logits (推理生成时用)。
                    原理: 避免在lm_head上进行无用的矩阵乘法计算。
        
        Returns:
            CausalLMOutputWithPast: 包含loss, logits, hidden_states, past_key_values, aux_loss
        """
        # ========== 步骤 1: 骨干网络特征提取 ==========
        # 数据流经Transformer的所有层
        # hidden_states: [Batch, Seq_Len, Hidden_Size]
        # past_key_values: 包含了当前步新生成的KV Cache
        # aux_loss: 如果使用了MoE，这里会返回负载均衡损失；否则为 0
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # ========== 步骤 2: Logits计算范围优化 (Logits Slicing) ==========
        # lm_head的计算量是 O(Seq_Len * Hidden * Vocab)，非常巨大。
        # 在推理时，我们只需要最后一个词的预测结果，不需要前文的预测。
        if isinstance(logits_to_keep, int):
            # 整数模式
            # logits_to_keep = 1 -> slice(-1, None) -> 取最后 1 个
            # logits_to_keep = 0 -> slice(None)     -> 取全部 (训练时)
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # 张量模式 (高级用法，指定特定位置)
            slice_indices = logits_to_keep
        
        # 对Hidden States进行切片，只保留需要计算的部分
        # 推理时: [Batch, 100, Hidden] -> [Batch, 1, Hidden]
        # 训练时: [Batch, 100, Hidden] -> [Batch, 100, Hidden]
        sliced_hidden_states = hidden_states[:, slice_indices, :]
        
        # ========== 步骤 3: 映射到词表 (Projection) ==========
        # 执行矩阵乘法: X @ W.T
        # logits形状: [Batch, Sliced_Len, Vocab_Size]
        # 这里的logits是未归一化的概率 (Log-odds)
        logits = self.lm_head(sliced_hidden_states)

        # ========== 步骤 4: 计算损失 (仅训练模式) ==========
        loss = None
        if labels is not None:
            # 因果语言模型的核心逻辑: "Shift Prediction" (位移预测)
            # 目标: 第t个时间步的Logit，应该预测第t+1个时间步的Label。
            # [Input]:  A  B  C  D
            # [Target]: B  C  D  E
            # shift_logits: 去掉最后一个Logit (因为它预测的是E，但Input只有到D)
            # 形状: [Batch, Seq_Len-1, Vocab]
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels: 去掉第一个Label (因为A之前没有Logit预测它)
            # 形状: [Batch, Seq_Len-1]
            shift_labels = labels[..., 1:].contiguous()
            # 计算交叉熵损失 (Cross Entropy)
            # view(-1): 将Batch和Seq维度展平，变成 [Total_Tokens, Vocab] 以适配Loss函数
            # ignore_index=-100: 忽略标签为 -100 (Padding) 的位置，不计算梯度
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100)
        
        # ========== 步骤 5: 封装输出 ==========
        # 使用Hugging Face标准格式返回，确保兼容性        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        # [MoE特有] 挂载辅助损失
        output.aux_loss = aux_loss
        return output