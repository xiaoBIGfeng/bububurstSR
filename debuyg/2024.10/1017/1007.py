import torch
import torch.nn as nn

# input = torch.rand(4,3,10,20)
# layer = nn.AdaptiveAvgPool2d((1, 1))
# out = layer(input)
# print(out.shape)

import torch
import torch.nn as nn
from torch.autograd import gradcheck

class SelectiveLayer(nn.Module):
    def __init__(self):
        super(SelectiveLayer, self).__init__()

    def forward(self, burst, v):
        mask = v.squeeze() == 1
        new_burst = burst[mask]
        return new_burst

# 创建示例数据
burst = torch.rand(14, 48, 64, 64, dtype=torch.double, requires_grad=True)
v = torch.randint(0, 2, (14, 1), dtype=torch.double)

# 确保输入数据是 double 类型
burst = burst.requires_grad_(True)
v = v.requires_grad_(False)  # 路由向量不需要梯度

# 实例化网络
selective_layer = SelectiveLayer()

# 使用 gradcheck 进行梯度检查
input = (burst, v)
test = gradcheck(selective_layer, input, eps=1e-6, atol=1e-4)

print("Gradient check passed:", test)
