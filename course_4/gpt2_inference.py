import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2TokenizerFast

# 加载模型配置
config = GPT2Config(
    vocab_size=50258,  # 词汇表大小要与训练时一致
    n_embd=768,  # 隐藏层大小
    n_layer=12,  # Transformer 层数
    n_head=12    # 注意力头的数量
)

# 初始化模型架构
class SimpleGPT2(nn.Module):
    def __init__(self, config):
        super(SimpleGPT2, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.n_layer)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, input_ids):
        input_ids = input_ids.long().to(self.embedding.weight.device)
        embeddings = self.embedding(input_ids)
        embeddings = embeddings.transpose(0, 1)
        embeddings = embeddings.to(self.decoder_layer.self_attn.in_proj_weight.device)
        output = self.transformer_decoder(embeddings, embeddings)
        return self.fc_out(output.transpose(0, 1))

# 加载模型权重
model = SimpleGPT2(config)
model.load_state_dict(torch.load('./GPT2_trained/pytorch_model.bin'))

# 将模型设置为推理模式
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval().to(device)

# 加载保存的 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('./GPT2_trained')

# 生成推理输入 (input_text 是用户输入的初始文本)
input_text = "Once upon a time"  # 例如，假设生成文本从这个初始语句开始
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

generated_text = input_text

# 设置生成长度
max_length = 50

# 生成 max_length 个 token
for _ in range(max_length):
    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(input_ids)

    # 获取最后一个时间步的输出，并选择最高概率的 token
    predicted_token_id = torch.argmax(outputs[:, -1, :], dim=-1).item()

    # 如果预测到结束 token，则停止生成
    if predicted_token_id == tokenizer.eos_token_id:
        break

    # 将预测的 token 解码并添加到生成文本中
    predicted_token = tokenizer.decode([predicted_token_id])
    generated_text += predicted_token

    # 将预测的 token 附加到输入中，继续生成下一个 token
    input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]]).to(input_ids.device)], dim=1)

# 打印生成的文本
print(f"Generated text: {generated_text}")
