import os
import time
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model import MiniGPTConfig, MiniGPTForCausalLM, apply_lora, load_lora
from trainer.utils import setup_seed, get_model_params

warnings.filterwarnings("ignore")

def init_model():
    assert os.path.exists(args.model_path), "模型权重文件不存在"
    _, model_name = os.path.split(args.model_path)
    print(f"加载模型权重文件: {args.model_path}")
    if model_name.endswith(".pth"):
        items = model_name[:-4].split("_")
        model = MiniGPTForCausalLM(MiniGPTConfig(
            hidden_size=int(items[1]),
            num_hidden_layers=int(items[2]),
            use_moe=True if "moe" in model_name else False,
            inference_rope_scaling=args.use_rope_scaling
        ))
        model_parameters = [n for n, _ in model.named_parameters()]
        ckp_weights = torch.load(args.model_path, map_location=args.device)["model"]
        # MiniGPTForCausalLM模型结构中embed_tokens和lm_head共享权重，因此命名参数比权重名称个数少一
        assert len(ckp_weights.keys()) == len(model_parameters) + 1, "权重文件与模型结构不匹配"
        model.load_state_dict(ckp_weights, strict=True)
        # 加载LoRA权重
        if os.path.exists(args.lora_path):
            print(f"加载LoRA权重文件: {args.lora_path}")
            apply_lora(model)
            load_lora(model, args.lora_path)
    else:
        model_name = "unknown"
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("./model")
    return model.eval().to(args.device), tokenizer, model_name

def main():
    parser = argparse.ArgumentParser(description="MiniGPT模型推理与对话")
    parser.add_argument("--model_path", type=str, default="./checkpoints/sft_768_16.pth", help="模型权重加载路径（默认sft）")
    parser.add_argument("--lora_path", type=str, default="none", help="LoRA权重加载路径")
    parser.add_argument("--use_rope_scaling", action="store_true", default=False,  help="是否启用RoPE位置编码外推（默认不启用）")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument("--temperature", type=float, default=0.85, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument("--top_p", type=float, default=0.85, help="nucleus采样阈值（0-1）")
    parser.add_argument("--historys", type=int, default=0, help="携带历史对话轮数（0表示不携带历史）")
    parser.add_argument("--show_speed", action="store_true", default=False, help="显示decode速度（默认不显示）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的？",
        "请用Python写一个计算斐波那契数列的函数",
        "解释一下'光合作用'的基本过程",
        "如果明天下雨，我应该如何出门？",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食"
    ]
    
    print("----------------------------")
    print("|   MiniGPT模型推理与对话   |")
    print("----------------------------")
    model, tokenizer, model_name = init_model()
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n请选择："))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("== 🤶 ==: "), '')
    conversation = []
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0: 
            print(f"== 🤶 ==: {prompt}")
        conversation = conversation[-2 * args.historys:] if args.historys > 0 else []
        conversation.append({"role": "user", "content": prompt})
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        inputs = (tokenizer.bos_token + prompt) if model_name.startswith("pretrain") else tokenizer.apply_chat_template(**templates)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
        print(f"== 🤖 ==: ", end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f"== 🚀 ==: [Speed] {gen_tokens / (time.time() - st):.2f} tokens/s\n") if args.show_speed else print("\n")

if __name__ == "__main__":
    main()
