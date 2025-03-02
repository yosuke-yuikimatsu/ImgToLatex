import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint

class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size, embed_dim=2176, num_heads=16, num_layers=8, ffn_dim=2048, max_length=300, sos_index=1, eos_index=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_length, embed_dim)
        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.vocab_size = vocab_size

    def forward(self, tgt_tokens, memory):
        device = memory.device
        if tgt_tokens is not None:
            B, T = tgt_tokens.shape
            tgt_emb = self.embedding(tgt_tokens[:,:-1]) + self.pos_encoding(torch.arange(T - 1, device=device))
            num_segments = 3
            segment_size = len(self.transformer_decoder) // num_segments
            x = tgt_emb
            for i in range(0, len(self.transformer_decoder), segment_size):
                x = checkpoint(self._forward_segment, x, memory, i, segment_size,use_reentrant=False)
            logits = self.output_layer(x)
            return logits
        else:
            B = memory.shape[0]
            generated_tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=device)
            logits_list = []
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for t in range(self.max_length - 1):
                if finished.all():
                    break
                tgt_emb = self.embedding(generated_tokens) + self.pos_encoding(torch.arange(generated_tokens.size(1), device=device))
                num_segments = 3
                segment_size = len(self.transformer_decoder) // num_segments
                x = tgt_emb
                for i in range(0, len(self.transformer_decoder), segment_size):
                    x = checkpoint(self._forward_segment, x, memory, i, segment_size,use_reentrant=False)
                output = x[:, -1, :]
                logits = self.output_layer(output)
                logits_list.append(logits.unsqueeze(1))
                next_token = logits.argmax(dim=-1, keepdim=True)
                finished = finished | (next_token.squeeze(-1) == self.eos_index)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            all_logits = torch.cat(logits_list, dim=1)
            predicted_tokens = generated_tokens
            max_len = predicted_tokens.size(1)
            mask = (predicted_tokens == self.eos_index).cumsum(dim=1) == 0
            mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1)
            lengths = mask.sum(dim=1)
            predicted_tokens = [predicted_tokens[i, :lengths[i]] for i in range(B)]
            all_logits = [all_logits[i, :lengths[i]-1] for i in range(B)]
            return all_logits, predicted_tokens

    def _forward_segment(self, x, memory, start, segment_size):
        for i in range(start, start + segment_size):
            x = self.transformer_decoder[i](x, memory)
        return x
