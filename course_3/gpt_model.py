# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


class GPTConfig:
    vocab_size: int = 16000  # 语料库的大小，即所有的token都在这里，token有16000个。
    seq_len: int = 128  # 预测的truck长度，即处理语料的窗口大小。
    d_model: int = 128  # d_model, token表征的embedding向量维度数。
    n_layer: int = 4  # decoder的层数量
    n_head: int = 4  # 多头注意力中头，即head的数量，每个head可以提取不同的特征。
    bias: bool = True  # 偏置值,即nn.Linear中y = ax + b 中的 b
    dropout: float = 0.0  # 防止过拟合的dropout层，即随机的扔掉dropout参数对应比例的神经元


class SinusoidPE(nn.Module):  # position embedding函数，实例化一个position embedding的表格，拿来就用。
    """ sin/cos position encoding """

    def __init__(self, config):
        super().__init__()
        d_model, seq_len = config.d_model, config.seq_len  # 载入config参数
        pe = torch.zeros(seq_len, d_model)  # 创建pe张量， 维度为seq_len * d_model
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # 用arrange函数创建0到seq_len的一维张量，
        # 然后用unsqueeze(1)在1的维度位置添加一个维度，变成 128 * 1 维向量。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 这里对pos的公式做
        # 了优化。对分母去ln对数，简化成： pos * exp( 2i/d_model * -ln(10000))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin实现
        pe[:, 1::2] = torch.cos(position * div_term)  # cos实现
        pe = pe.unsqueeze(0)
        # 前面加一个维度，让position embedding 变成 1 * C * E -> 以后训练是导入Batch个C * E一起训练的，
        # 所以现在把它展开成 1 * C * E 的维度，未来训练的时候，利用torch的广播功能，能过让所有的Batch获得position embedding
        self.register_buffer('sinusoid_pe', pe)  # 我们的position embedding是不参与训练的，所以这里可以将其注册到buffer
        # 使得后续计算更加快速，并节省内存空间


    # <cls> <eos> <pad>
    def forward(self, x):
        return self.sinusoid_pe[:, :x.shape[1], :] # 在构造函数init里已经实现了position embedding，所以这里仅仅
        # 把构造的函数return即可


class SelfAttention(nn.Module):
    """ multi-head attention """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0  #
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len))
                             .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x):
        B, C, E = x.size()      # 获取x的3个维度，分别是Batch, Context, Embedding

        q, k, v = self.attn(x).split(self.d_model, dim=2)       #  计算Q, K, V, 并在最后一维上做切分。
        # 切分成d_model维度。本质上这里是并行计算
        q = q.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)  # (B, nh, C, hs)
        k = k.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)  # (B, nh, C, hs)
        v = v.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)  # (B, nh, C, hs)

        # self-attention: (B, nh, C, hs) x (B, nh, hs, C) -> (B, nh, C, C)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :C, :C] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, C, C) x (B, nh, C, hs) -> (B, nh, C, hs)
        y = y.transpose(1, 2).contiguous().view(B, C, E)

        return self.proj(y)


class FeedFoward(nn.Module):  # 前馈神经网络FFN
    """ a two-layers mlp """
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 神经元升维
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # 神经元降维
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Decoder Block """

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)

        self.ffn = FeedFoward(config)           # switch to KAN

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        # x = x + self.attn(x)
        # x =  self.ffn(self.ln2(x))
        x = x + self.ffn(x)
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed_table = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed_table = SinusoidPE(config)
        self.decoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features, targets=None):
        tok_emb = self.tok_embed_table(features)  # B, C, E
        pos_emb = self.pos_embed_table(tok_emb)  # 1, C, E
        x = tok_emb + pos_emb   # B, C, E
        x = self.decoder_blocks(x)  # B, C, E

        x = self.layer_norm(x)
        if targets is not None:
            logits = self.final_linear(x)  # (B,C,V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.final_linear(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, seq, max_new_tokens):
        for _ in range(max_new_tokens):
            seq = seq[:, -self.config.seq_len:]
            logits, _ = self(seq)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)
            # sample from the distribution
            seq_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            seq = torch.cat((seq, seq_next), dim=1)
        return seq


def main():
    config = GPTConfig()
    model = GPTModel(config)
    summary(model, input_size=[(100, config.seq_len), (100, config.seq_len)],
            dtypes=[torch.long, torch.long])


if __name__ == '__main__':
    main()
