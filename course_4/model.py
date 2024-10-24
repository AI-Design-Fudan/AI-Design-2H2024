# model.py
import torch
from torch import nn

class SimpleGPT2(nn.Module):
    def __init__(self, config):
        super(SimpleGPT2, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        embeddings = embeddings.transpose(0, 1)

        # 创建因果掩码
        seq_length = input_ids.size(1)
        device = input_ids.device
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()

        # 应用掩码
        hidden_states = self.transformer_decoder(embeddings, embeddings, tgt_mask=causal_mask)
        hidden_states = self.ln_f(hidden_states)
        logits = self.fc_out(hidden_states.transpose(0, 1))
        return logits
