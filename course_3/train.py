from gpt_model import GPTConfig, GPTModel
import numpy as np
import sentencepiece as spm
import sys
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # 引入 tqdm 库

device = 'cuda' if torch.cuda.is_available() else 'mps'

learning_rate = 1e-3
max_iters = 5000

train_data = np.memmap('./train.dat', dtype=np.int32, mode='r')
test_data = np.memmap('./test.dat', dtype=np.int32, mode='r')


def get_batch(split, config):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - config.seq_len, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + config.seq_len]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config.seq_len]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train(config, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    # 创建一个实时绘制的图形
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], label='Training Loss')
    plt.legend()

    # 使用 tqdm 创建进度条
    with tqdm(total=max_iters, desc="Training Progress", ncols=100) as pbar:
        for iter_num in range(max_iters):
            optimizer.zero_grad()

            xb, yb = get_batch('train', config)

            # forward and loss calculation
            _, loss = model(xb, yb)
            losses.append(loss.item())

            if (iter_num + 1) % 100 == 0:
                print(f"[train_info] iter:{iter_num + 1:5d}, loss:{loss.item():5.3f}")

                # 更新图形
                line.set_data(range(len(losses)), losses)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

            # backward and gradient descent
            loss.backward()

            # update weights
            optimizer.step()

            # 更新进度条
            pbar.update(1)

    plt.ioff()  # 关闭实时模式
    plt.show()  # 显示最终图形
    print(f"final loss: {loss.item()}")


def main():
    config = GPTConfig()
    config.batch_size = 32
    config.dropout = 0.1

    model = GPTModel(config).to(device)
    # load tokenizer
    from data_set import load_tokenizer

    model_file = "tale.model"
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)

    # 文本生成部分的可视化
    user_inputs = ["郭靖一掌挥出", "黄蓉突然想到", "周伯通好奇心大起", "洪七公哈哈大笑", "方文山大醉"]
    for user_input in user_inputs:
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        print(f"输入: {user_input}")
        gpt_output = model.generate(context, max_new_tokens=50)[0].tolist()

        # 逐步展示生成文本
        for token_id in gpt_output:
            generated_text = sp.decode([token_id])
            print(generated_text, end='', flush=True)
            time.sleep(0.1)
        print("\n" + "="*50)

    train(config, model)  # 开始训练

    # 训练结束后再生成一组文本
    user_inputs = ["郭靖一掌挥出", "黄蓉突然想到", "周伯通好奇心大起", "洪七公哈哈大笑", "方文山大醉"]
    for user_input in user_inputs:
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        print(f"输入: {user_input}")
        gpt_output = model.generate(context, max_new_tokens=50)[0].tolist()

        # 逐步展示生成文本
        for token_id in gpt_output:
            generated_text = sp.decode([token_id])
            print(generated_text, end='', flush=True)
            time.sleep(0.1)
        print("\n" + "="*50)


if __name__ == '__main__':
    main()