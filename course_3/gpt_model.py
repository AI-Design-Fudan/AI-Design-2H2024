import math
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary


class GPTConfig:
    vocab_size: int = 16000
    seq_len: int = 128
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    bias: bool = True
    dropout: float = 0.0


class SinusoidPE(nn.Module):
    """ sin/cos position encoding """

    def __init__(self, config):
        super().__init__()
        d_model, seq_len = config.d_model, config.seq_len
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('sinusoid_pe', pe)

    def forward(self, x):
        return self.sinusoid_pe[:, :x.shape[1], :]


class SelfAttention(nn.Module):
    """ multi-head attention """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len))
                             .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x, visualize_attention=False):  # 新增参数：visualize_attention
        B, C, E = x.size()
        q, k, v = self.attn(x).split(self.d_model, dim=2)
        q = q.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)
        k = k.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)
        v = v.view(B, C, self.n_head, E // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :C, :C] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # 可视化注意力矩阵，但仅在需要时
        if visualize_attention:
            plt.figure(figsize=(10, 8))
            sns.heatmap(att[0][0].cpu().detach().numpy(), cmap="viridis")
            plt.title("Self-Attention Heatmap")
            plt.show()

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, C, E)

        return self.proj(y)


class FeedFoward(nn.Module):
    """ a two-layers mlp """
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Decoder Block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedFoward(config)

    def forward(self, x, visualize_attention=False):
        x = self.ln1(x)
        x = x + self.attn(x, visualize_attention=visualize_attention)    #
        x = self.ln2(x)
        x = x + self.ffn()                                              #
        # x = x + self.ffn(self.ln2(x))                                              #
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

    def forward(self, features, targets=None, visualize_attention=False):
        tok_emb = self.tok_embed_table(features)  # B, C, E
        pos_emb = self.pos_embed_table(tok_emb)  # 1, C, E
        x = tok_emb + pos_emb   # B, C, E
        for block in self.decoder_blocks:
            x = block(x, visualize_attention=visualize_attention)  # 添加 visualize_attention 传递

        x = self.layer_norm(x)
        logits = self.final_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, seq, max_new_tokens, visualize_attention=False):
        for _ in range(max_new_tokens):
            seq = seq[:, -self.config.seq_len:]
            logits, _ = self.forward(seq, visualize_attention=visualize_attention)  # 调用forward时传递 visualize_attention 参数
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            seq_next = torch.multinomial(probs, num_samples=1)
            seq = torch.cat((seq, seq_next), dim=1)
        return seq


def main():
    config = GPTConfig()
    model = GPTModel(config)
    summary(model, input_size=[(100, config.seq_len), (100, config.seq_len)],
            dtypes=[torch.long, torch.long])


if __name__ == '__main__':
    main()