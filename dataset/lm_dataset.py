import os
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minigpt，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minigpt，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minigpt, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minigpt, a small but useful language model."
    ]
    # 第一条对话的role不为system，需要按照一定的概率从SYSTEM_PROMPTS随机挑选一条，作为系统提示词
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    # 按照一定概率移除文本中的 “<think>\n\n</think>\n\n” 标识符
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", '')
    return prompt_content

class PretrainDataset(Dataset):
    """预训练数据集
    数据格式：
    {"text": "鉴别一组中文文章的风格和特点，例如官方、口语、文言等..."}
    {"text": "请从这首诗选出三句话，重新组成另一首新的诗歌..."}
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用Hugging Face的datasets库加载本地JSONL文件
        # 此时数据被加载为内存映射对象，self.samples[index] 会返回一个字典
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 根据索引从数据集提取出 "text" 字段的内容
        # 将文本转为数字ID，预留2个位置给 BOS 和 EOS
        # truncation=True 保证长度不超过 max_length - 2
        tokens = self.tokenizer(str(sample["text"]), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 在文本前后分别加上“开始符”(BOS)和“结束符”(EOS)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 如果长度不足max_length，在后面补齐pad_token_id，最终得到一个长度固定为max_length的列表
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 将Python列表转为PyTorch的长整型张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 克隆一份input_ids作为训练的目标（预测下一个词）
        labels = input_ids.clone()
        # 将所有Padding部分的标签设为 -100，在计算CrossEntropyLoss时，将忽略这些位置，不计算Loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

class SFTDataset(Dataset):
    """监督微调数据集
    数据格式：
    {"conversations": [
        {"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"},
        {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针..."}]
    }
    {"conversations": [
        {"role": "user", "content": "请描述一下北京的春天。"},
        {"role": "assistant", "content": "北京的春天是一年四季中最令人期待的季节之一..."}]
    }
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用Hugging Face的load_dataset加载JSONL。
        # 它的优势是Lazy Loading（懒加载）和Memory Mapping（内存映射）。
        # 即便jsonl文件有几GB，也不会一次性读入内存，而是用到哪条读哪条。        
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 为了实现“只对Assistant的回复计算Loss”，我们需要在长文本中找到回复的起始和结束位置。
        # 这里预先计算好“Assistant起始符”和“结束符”对应的Token ID序列。
        # 注意：add_special_tokens=False 很重要，因为我们不想要BOS/EOS再次包裹这些片段。
        # 假设模板是ChatML，这里硬编码了 'assistant\n'。
        # 风险提示：如果你的Chat Template渲染出来是 '<|im_start|>assistant' (没换行)，
        # 这里的匹配逻辑就会失效，导致Loss全为 0。
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        # 兼容简单的工具调用（如果有function call）
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        # 使用Tokenizer自带的apply_chat_template进行渲染
        return self.tokenizer.apply_chat_template(
            messages,
            # 关键：这里只拼成字符串，先不转ID，方便后续统一截断和处理
            tokenize=False,
            # SFT是训练已有对话，不需要像推理时那样自动添加 "assistant:" 引导头
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """Loss Masking / Label Masking
        这是SFT代码的灵魂，生成与input_ids等长的labels序列。
        规则：
        - User 的话 -> 设为 -100 (PyTorch CrossEntropyLoss 默认忽略 -100)
        - Assistant 的话 -> 设为原本的Token ID (参与计算梯度)
        - Padding -> 设为 -100
        """
        # 先初始化全为 -100
        labels = [-100] * len(input_ids)
        i = 0
        # 线性扫描input_ids，寻找Assistant的回复区间
        while i < len(input_ids):
            # 判断当前位置是否匹配“Assistant起始符”：<|im_start|>assistant\n
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 继续向后找，直到找到“结束符”：<|im_end|>\n
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将start到end之间的部分（即回复内容）从 -100 恢复为真实的input_ids
                # 只有这一部分会产生梯度，更新模型参数
                # min(..., self.max_length) 是防止越界                
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 移动指针i到当前回复结束的位置，继续找下一轮（多轮对话场景）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果没匹配到，指针后移一位
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 提示词前处理：按概率增加system部分
        conversations = pre_processing_chat(sample["conversations"])
        # 将prompt由字典形式转换为长字符串形式
        prompt = self.create_chat_prompt(conversations)
        # 提示词后处理：按概率移除<think>标识符（训练推理模型时需要）
        prompt = post_processing_chat(prompt)
        # 转换为Token IDs，并截断多余的部分
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 如果长度不够max_length，用pad_token_id补齐
        # 这样做是为了让一个Batch内的数据维度一致，才能堆叠成Tensor
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成Mask后的标签
        labels = self.generate_labels(input_ids)

        # # ===== 调试代码 (强烈建议在正式训练前取消注释跑一次) =====
        # # 作用：人工肉眼检查Mask是否正确，也就是User部分对应的Label是否是-100。
        # print(f"\n----- Sample {index} -----")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # =====================================================        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class DPODataset(Dataset):
    """RLHF强化学习数据集
    数据格式：                                                                     
    {"chosen": [
        {"role": "user", "content": "Find the size of angle x in the figure."}, 
        {"role": "assistant", "content": "To determine the size of angle x in the provided figure..."}],
    "rejected": [
        {"role": "user", "content": "Find the size of angle x in the figure."}, 
        {"role": "assistant", "content": "To find the size of angle x, let us apply the basic geometry..."}]
    }
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取padding token的id，如果没有则默认为 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 获取chosen (好回答)的对话列表
        chosen = sample["chosen"]
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        # 获取rejected (坏回答) 的对话列表
        rejected = sample["rejected"]
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 构造训练数据 (Shift Trick)
        # 在因果语言模型（Causal LM）训练中，我们预测下一个token。
        # 输入是x (0 到 N-1)，目标是y (1 到 N)，掩模mask和目标y是对应的
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        # 初始化全 0 的mask
        loss_mask = [0] * len(input_ids)
        i = 0
        # 遍历整个token序列
        while i < len(input_ids):
            # 寻找Assistant回答的“开始标记” (例如 "<|im_start|>assistant\n")
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 寻找Assistant回答的“结束标记” (例如 "<|im_end|>\n")
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将Start到End之间的部分mask设为 1，这部分就是模型实际生成的回答，我们需要计算它的Loss
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

class RLAIFDataset(Dataset):
    """RLAIF强化学习数据集
    数据格式：
    {"conversations": [
        {"role": "user", "content": "列出五个基本的人格理论，并分别以一句话概括。"}, 
        {"role": "assistant", "content": "空"}]
    }
    {"conversations": [
        {"role": "user", "content": "仔细阅读以下句子并回答“汤姆是医生还是建筑工人?”"}, 
        {"role": "assistant", "content": "空"}]
    }
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # add_special_tokens=False 表示不要自动在首尾添加额外的特殊符
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 用于存放标准格式的对话列表 (形如 [{"role": "user", "content": "..."}, ...])
        messages = []
        # 用于暂存最终的回复内容
        answer = ''
        for i, turn in enumerate(conversations):
            # 根据索引的奇偶性判断角色，偶数轮(0, 2, 4...)是user，奇数轮(1, 3, 5...)是assistant
            # 这种写法强制假设对话是user和assistant严格交替进行的
            role = "user" if i % 2 == 0 else "assistant"
            # 按照HuggingFace chat_template要求的格式拼接字典，并加入messages列表
            messages.append({"role": role, "content": turn["content"]})
            # 不断覆盖answer，当循环结束时，answer保存的就是对话列表中的最后一句话
            answer = turn["content"]
        # 使用tokenizer的chat template功能将历史对话格式化为单个字符串
        prompt = self.tokenizer.apply_chat_template(
            # 取除了最后一句话之外的所有对话（即历史上下文作为Prompt）
            messages[:-1],
            # 返回字符串，而不是token IDs，在后面训练阶段，做了padding后再tokenize
            tokenize=False,
            # 在字符串末尾自动加上让模型开始生成的引导符（如 "<|im_start|>assistant\n"）
            add_generation_prompt=True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])
        return {
            "prompt": prompt,
            "answer": answer
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    train_ds = SFTDataset("sft_mini_512.jsonl", tokenizer, max_length=340)
    print(train_ds[0])