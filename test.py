import torch
print(torch.__version__)  # 应该输出 2.2.2
print(torch.cuda.is_available()) # 应该输出 False (因为是CPU版本)