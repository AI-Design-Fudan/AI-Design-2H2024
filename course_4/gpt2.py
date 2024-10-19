import torch
import os
from torch import nn
from transformers import GPT2Config, AdamW, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn.functional as F

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
    n_layer=12,        # Transformer 层数
    n_head=12          # 注意力头的数量
)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def top_k_logits(logits, k):
    if k == 0:
        # 不进行截断
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :] / temperature
        filtered_logits = top_k_logits(next_token_logits, top_k)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# 从零开始创建模型
class SimpleGPT2(nn.Module):
    def __init__(self, config):
        super(SimpleGPT2, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)  # 初始化位置编码
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=0.1,
                activation='gelu'
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, input_ids):
        input_ids = input_ids.long()
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        seq_length = embeddings.size(0)
        mask = generate_square_subsequent_mask(seq_length).to(embeddings.device)

        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_mask=mask)

        hidden_states = self.ln_f(hidden_states)
        logits = self.fc_out(hidden_states.transpose(0, 1))
        return logits  # 返回形状为 (batch_size, seq_len, vocab_size)

# 使用本地 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 添加 pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')  # 设置 pad_token_id 为新添加的 token 的 ID

# 更新配置的 vocab_size
config.vocab_size = len(tokenizer)

# 初始化模型
model = SimpleGPT2(config).to(device)

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
    "num_epochs": 5,
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

        if step == 0:
            print(
                f"input_ids range: max={input_ids.max().item()}, min={input_ids.min().item()}, vocab_size={config.vocab_size}")

        if input_ids.max().item() >= config.vocab_size or input_ids.min().item() < 0:
            raise ValueError("input_ids 包含超出词汇表范围的值！")

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

# 示例：生成文本
prompt = "Once upon a time"
generated_text = generate_text(model, tokenizer, prompt)
print(f"Generated Text:\n{generated_text}")
