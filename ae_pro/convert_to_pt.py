import torch
import os
import modules_ae  # 确保你引入了正确定义的模型类
from modules_ae import AELightning

ckpt_path= "/root/model_ae/lightning_logs/version_0/checkpoints/last.ckpt"
model = AELightning.load_from_checkpoint(ckpt_path)

# 提取 PyTorch 模型本体（即 self.model）
torch_model = model.model

# 自动保存在 submit 目录
save_dir = "submit"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "your_model.pt")

# 保存模型参数
torch.save(torch_model.state_dict(), save_path)

print(f"✅ .pt 文件已保存：{save_path}")
