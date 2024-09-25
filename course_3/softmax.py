import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, interactive


def softmax(x, temperature=1.0):
    e_x = np.exp(x / temperature - np.max(x / temperature))
    return e_x / e_x.sum(axis=0)


# 随机生成 scores
np.random.seed(0)
scores = np.random.randn(6)


def update_plot(temperature):
    probabilities = softmax(scores, temperature)

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.bar(range(len(scores)), scores, color='blue')
    plt.title('Scores')
    plt.xticks(range(len(scores)))

    plt.subplot(1, 2, 2)
    plt.bar(range(len(probabilities)), probabilities, color='orange')
    plt.title(f'Softmax Probabilities (Temperature={temperature})')
    plt.xticks(range(len(probabilities)))

    plt.show()


# 创建一个滑动条
temperature_slider = FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Temperature:')
interactive_plot = interactive(update_plot, temperature=temperature_slider)

# 显示交互式图形
interactive_plot
