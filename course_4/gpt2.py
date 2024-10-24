import torch
import os
from torch import nn
from transformers import GPT2Config, AdamW, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 导入 SimpleGPT2 类
from model import SimpleGPT2

# 检查 CUDA 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置 CUDA_LAUNCH_BLOCKING=1 以调试 CUDA 错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 加载 wikitext-2-v1 数据集
dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')

# 加载 GPT-2 配置
config = GPT2Config(
    vocab_size=50258,  # GPT-2 的词汇表大小
    n_positions=1024,  # 设置位置编码的最大长度
    n_embd=768,        # 隐藏层大小
    n_layer=6,        # Transformer 层数
    n_head=12          # 注意力头的数量
)

# 从零开始创建模型
model = SimpleGPT2(config).to(device)

# 使用本地 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 添加 pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')  # 设置 pad_token_id 为新添加的 token 的 ID

# 更新配置的 vocab_size
config.vocab_size = len(tokenizer)

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
    "num_epochs": 2,
}

# 创建 DataLoader
train_loader = DataLoader(tokenized_datasets, batch_size=train_args['batch_size'], shuffle=True, collate_fn=collate_fn)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=train_args['learning_rate'])

# 训练模型
model.train()
for epoch in range(train_args['num_epochs']):
    total_loss = 0
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
output_dir = "./GPT2_trained"
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)

print("模型训练完成并已保存。")
