# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import GPT2Config, GPT2TokenizerFast
#
# # 加载保存的 tokenizer
# tokenizer = GPT2TokenizerFast.from_pretrained('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained')
#
# # 加载保存的配置
# config = GPT2Config.from_pretrained('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained')
#
# # 修改后的 SimpleGPT2 类（添加因果掩码）
# class SimpleGPT2(nn.Module):
#     def __init__(self, config):
#         super(SimpleGPT2, self).__init__()
#         self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
#         self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
#         nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
#
#         self.decoder_layer = nn.TransformerDecoderLayer(
#             d_model=config.n_embd,
#             nhead=config.n_head,
#             dim_feedforward=4 * config.n_embd,
#             dropout=0.1,
#             activation='gelu'
#         )
#         self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.n_layer)
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.fc_out = nn.Linear(config.n_embd, config.vocab_size)
#
#     def forward(self, input_ids):
#         embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
#         embeddings = embeddings.transpose(0, 1)
#
#         # 创建因果掩码
#         seq_length = input_ids.size(1)
#         device = input_ids.device
#         causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()
#
#         # 应用掩码
#         hidden_states = self.transformer_decoder(embeddings, embeddings, tgt_mask=causal_mask)
#         hidden_states = self.ln_f(hidden_states)
#         logits = self.fc_out(hidden_states.transpose(0, 1))
#         return logits
#
# # 加载模型权重
# model = SimpleGPT2(config)
# model.load_state_dict(torch.load('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained\\pytorch_model'
#                                  '.bin'))
#
# # 将模型设置为推理模式
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.eval().to(device)
#
# # 生成推理输入
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
#
# generated_text = input_text
#
# # 设置生成长度和采样参数
# max_length = 50
# temperature = 1.0
# top_k = 50
#
# def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
#     """过滤 logits 以仅保留 top-k 和/或 nucleus（top-p） tokens"""
#     assert logits.dim() == 2  # Batch size x Vocabulary size
#
#     top_k = min(top_k, logits.size(-1))  # 防止 top_k 超过词汇表大小
#
#     if top_k > 0:
#         # 移除 top_k 之后的 tokens
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
#         logits[indices_to_remove] = filter_value
#
#     if top_p > 0.0:
#         # 计算累积概率
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#         # 移除累积概率超过 top_p 的 tokens
#         sorted_indices_to_remove = cumulative_probs > top_p
#         # 保证至少有一个词保留
#         sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
#         sorted_indices_to_remove[:, 0] = 0
#
#         # 将需要移除的 tokens 的 logits 设置为 filter_value
#         indices_to_remove = sorted_indices[sorted_indices_to_remove]
#         logits.scatter_(1, indices_to_remove, filter_value)
#
#     return logits
#
# # 生成 max_length 个 token
# # 生成 max_length 个 token
# for _ in range(max_length):
#     with torch.no_grad():
#         # 截断 input_ids，确保长度不超过 n_positions
#         if input_ids.size(1) > config.n_positions:
#             input_ids = input_ids[:, -config.n_positions:]
#
#         outputs = model(input_ids)
#         next_token_logits = outputs[:, -1, :] / temperature
#
#         # 直接从 logits 中采样
#         probabilities = F.softmax(next_token_logits, dim=-1)
#         predicted_token_id = torch.multinomial(probabilities, num_samples=1).item()
#
#     # 如果预测到结束标记，则停止生成
#     if predicted_token_id == tokenizer.eos_token_id:
#         break
#
#     # 将预测的 token 解码并添加到生成文本中
#     predicted_token = tokenizer.decode([predicted_token_id])
#     generated_text += predicted_token
#
#     # 将预测的 token 附加到输入中，继续生成下一个 token
#     input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]]).to(device)], dim=1)
#
# # 打印生成的文本
# print(f"Generated text: {generated_text}")
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2TokenizerFast

# 导入 SimpleGPT2 类
from model import SimpleGPT2

# 加载保存的 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained')

# 加载保存的配置
config = GPT2Config.from_pretrained('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained')

# 加载模型权重
model = SimpleGPT2(config)
model.load_state_dict(torch.load('E:\\pythonProject\\2H2024\\AI-Design-2H2024\\course_4\\GPT2_trained\\pytorch_model.bin'))

# 将模型设置为推理模式
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# 生成推理输入
input_text = "The game began "
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

generated_text = input_text

# 设置生成长度和采样参数
max_length = 50
temperature = 1.0
top_k = 50

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """过滤 logits 以仅保留 top-k 和/或 nucleus（top-p） tokens"""
    assert logits.dim() == 2  # Batch size x Vocabulary size

    top_k = min(top_k, logits.size(-1))  # 防止 top_k 超过词汇表大小

    if top_k > 0:
        # 取 top_k 个最大值，其余设置为 filter_value
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # 计算累积概率
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过 top_p 的 tokens
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保证至少有一个词保留
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # 将需要移除的 tokens 的 logits 设置为 filter_value
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits.scatter_(1, indices_to_remove, filter_value)

    return logits

# 生成 max_length 个 token
for _ in range(max_length):
    with torch.no_grad():
        # 截断 input_ids，确保长度不超过 n_positions
        if input_ids.size(1) > config.n_positions:
            input_ids = input_ids[:, -config.n_positions:]

        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :] / temperature

        # 应用 top_k 过滤
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=0.0)

        # 从过滤后的 logits 中采样
        probabilities = F.softmax(next_token_logits, dim=-1)
        predicted_token_id = torch.multinomial(probabilities, num_samples=1).item()

    # 如果预测到结束标记，则停止生成
    if predicted_token_id == tokenizer.eos_token_id:
        break

    # 将预测的 token 解码并添加到生成文本中
    predicted_token = tokenizer.decode([predicted_token_id])
    generated_text += predicted_token

    # 将预测的 token 附加到输入中，继续生成下一个 token
    input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]]).to(device)], dim=1)

# 打印生成的文本
print(f"Generated text: {generated_text}")
