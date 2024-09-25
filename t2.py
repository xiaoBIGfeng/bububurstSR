import torch

# 假设你有一个形状为[1, 14, 3, 80, 80]的RGB张量，其中14是批量大小
# 这里我们随机生成一个这样的张量作为示例
rgb_tensor = torch.rand(1, 14, 3, 80, 80)

# 选择要复制的通道，例如复制绿色通道
# 我们可以通过索引来选择绿色通道，并将其复制到新的通道
green_channel = rgb_tensor[:, :, 1:2, :, :]  # 选择绿色通道，形状为[1, 14, 1, 80, 80]

# 复制绿色通道来创建RGGB格式
rggb_tensor = torch.cat((rgb_tensor, green_channel), dim=2)  # dim=2是通道维度

# 现在rggb_tensor的形状应该是[1, 14, 4, 80, 80]
print(rggb_tensor.shape)
