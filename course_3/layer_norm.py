import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
data = np.random.rand(10, 10) * 10  # 10 行 10 列

# 设置两行特征，使它们相差较大
data[0] = [1, 2, 1, 1, 2, 1, 1, 2, 1, 1]  # 特征1
data[1] = [5, 6, 5, 5, 6, 5, 5, 6, 5, 5]  # 特征2

# 其余行保持随机
data[2:] = np.random.rand(8, 10) * 10

# 计算 LayerNorm
def layer_norm(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + 1e-6)

normalized_data = layer_norm(data)

# 绘制热图
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 原始数据热图
cax1 = ax[0].matshow(data, cmap='viridis')
ax[0].set_title('Original Data')
plt.colorbar(cax1, ax=ax[0])

# LayerNorm 后的数据热图
cax2 = ax[1].matshow(normalized_data, cmap='viridis')
ax[1].set_title('LayerNorm Normalized Data')
plt.colorbar(cax2, ax=ax[1])

plt.show()
