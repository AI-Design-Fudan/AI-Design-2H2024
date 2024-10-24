import torch
import os
from torch import nn
from transformers import GPT2Config, AdamW, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 导入模型
from model import SimpleGPT2

# 检查 CUDA 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置 CUDA_LAUNCH_BLOCKING=1 以调试 CUDA 错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义保存模型的目录
output_dir = "./GPT2_trained"

# 加载保存的 tokenizer 和配置文件
tokenizer = GPT2TokenizerFast.from_pretrained(output_dir)
config = GPT2Config.from_pretrained(output_dir)

# 创建模型实例并加载保存的状态字典
model = SimpleGPT2(config).to(device)
model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))

# 准备数据集
dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')

# 准备数据
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# 进行数据预处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 移除 'text' 列
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 创建一个 collate 函数
def collate_fn(batch):
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {
        'input_ids': input_ids_padded.to(device),
    }

# 设置训练参数
train_args = {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 10,  # 继续训练一个 epoch
}

# 创建 DataLoader
train_loader = DataLoader(tokenized_datasets, batch_size=train_args['batch_size'], shuffle=True, collate_fn=collate_fn)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=train_args['learning_rate'])

# 开始继续训练
model.train()
total_loss = 0
epoch = 0  # 从第 1 个 epoch 开始
print(f"Starting epoch {epoch + 1}")
for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
    input_ids = batch['input_ids']

    # 前向传播
    outputs = model(input_ids)

    # 将 labels 右移一位
    labels = input_ids[:, 1:].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # 使用 -100 来忽略这些位置的损失计算

    # 调整 outputs，使其与 labels 对齐
    outputs = outputs[:, :-1, :]

    # 计算损失
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    # 每 5 步输出一次 loss
    if step % 5 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

avg_loss = total_loss / len(train_loader)
print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss}")

# 保存模型
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)

print("模型训练完成并已保存。")
