import torch

class LoRA(torch.nn.Module):
    """低秩自适应
    核心原理：预训练好的大模型在做下游任务的微调时，其权重改变量是低秩的。
    我们可以通过训练Dense层的秩分解矩阵，来间接实现Dense层的训练，从而在保证模型微调效果
    的前提下，显著降低训练参数量。

    在Pytorch框架下，假设某个Dense层的预训练权重矩阵尺寸是o*i，其中i是输入尺寸，o是输出尺寸。
    LoRA的做法是给Dense层增加一个旁路，引入降维矩阵A（尺寸为r*i）和升维矩阵B（尺寸为o*r），
    在训练过程中，保持预训练权重矩阵不变，只更新矩阵A和B的值，输出结果叠加原始权重矩阵和BA。
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank # 矩阵的秩，一般取值1,2,4,8，rank的值远小于in_features和out_features
        self.A = torch.nn.Linear(in_features, rank, bias=False)  # 降维矩阵A，尺寸为r*i
        self.B = torch.nn.Linear(rank, out_features, bias=False)  # 升维矩阵B，尺寸为o*r
        # 矩阵A随机高斯分布初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全零初始化，保证训练开始时BA仍然是零矩阵
        self.B.weight.data.zero_()

    def forward(self, x):
        # Pytorch线性变化公式：y=x*WT + b
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # LoRA本身不要求Linear层的in_features和out_features相等。原作者设置这里相等的含义未知？
        if isinstance(module, torch.nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward
            # 原始权重矩阵计算结果和LoRA计算结果叠加后输出，类似于残差连接
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {k.replace(f"{name}.lora.", ''): v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)