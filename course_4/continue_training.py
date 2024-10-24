import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import GPT, GPTConfig

# 使用 Hugging Face 的 datasets 库加载 wikitext-2 数据集
from datasets import load_dataset
from transformers import GPT2Tokenizer

# 导入 tqdm 库
from tqdm import tqdm

# 加载 GPT-2 分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 定义数据集类
class WikiTextDataset(Dataset):
    def __init__(self, texts, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.examples = []

        # 将所有文本连接在一起，然后进行分块
        full_text = "\n\n".join(texts)
        tokens = tokenizer.encode(full_text)

        # 创建输入和目标序列
        for i in range(0, len(tokens) - block_size, block_size):
            x = tokens[i:i + block_size]
            y = tokens[i + 1:i + block_size + 1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 加载 wikitext-2 数据集
dataset = load_dataset('wikitext', 'wikitext-103-v1')
train_texts = dataset['train']['text']

# 创建数据集和数据加载器
block_size = 1024  # 与之前的训练保持一致
train_dataset = WikiTextDataset(train_texts, block_size, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 根据内存情况调整 batch_size

# 初始化模型配置，使用与之前相同的参数
config = GPTConfig(
    vocab_size=50304,     # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)
model = GPT(config)

# 加载之前训练的模型权重
model_load_path = './GPT2_trained/trained_gpt_model_final.pth'
model.load_state_dict(torch.load(model_load_path))

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 创建保存模型的目录
save_dir = './GPT2_trained'
os.makedirs(save_dir, exist_ok=True)

# 继续训练循环
epochs = 1  # 您可以根据需要调整训练轮数
model.train()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
    for batch_idx, (x, y) in progress_bar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条描述
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=avg_loss)


# 最后保存最终模型
final_model_save_path = os.path.join(save_dir, 'trained_gpt_model_final.pth')
torch.save(model.state_dict(), final_model_save_path)
print(f"最终模型已保存到 '{final_model_save_path}'")
