import numpy as np
import matplotlib.pyplot as plt

# 获取位置编码的函数
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2*i] = np.sin(k / denominator)      # 偶数维度使用 sin
            P[k, 2*i+1] = np.cos(k / denominator)    # 奇数维度使用 cos
    return P

# 可视化 sin 函数的函数
def plotSinusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    plt.plot(x, y)
    plt.title('k = ' + str(k))

# 测试位置编码函数
P = getPositionEncoding(seq_len=4, d=4, n=100)
print(P)

# 绘制每个维度的 sin 曲线
fig = plt.figure(figsize=(15, 4))
for i in range(4):
    plt.subplot(141 + i)
    plotSinusoid(i*4)

seq_len = 1000
d = 512
n = 10000

# 生成大规模的位置信息并绘制矩阵
P = getPositionEncoding(seq_len=seq_len, d=d, n=n)
cax = plt.matshow(P, cmap='viridis')  # 热图显示
plt.gcf().colorbar(cax)
plt.show()
