import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size = 131, embed_dim=1536, num_heads=12, num_layers=12, ffn_dim=4096, max_length=50, sos_index=1, eos_index=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_length, embed_dim)
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.vocab_size = vocab_size

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, tgt_tokens, memory, beam_width=None):
        device = memory.device
        if tgt_tokens is not None:
            # Режим обучения
            B, T = tgt_tokens.shape
            T_minus_1 = T - 1
            tgt_emb = self.embedding(tgt_tokens[:, :-1]) + self.pos_encoding(torch.arange(T_minus_1, device=device))
            tgt_mask = self.generate_square_subsequent_mask(T_minus_1, device=device)
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output_layer(output)  # (B, T-1, vocab_size)
            return logits
        else:
            # Режим инференса
            if beam_width is None or beam_width == 1:
                # Жадный подход (оригинальный)
                return self._greedy_decode(memory)
            else:
                # Beam search
                return self._beam_search_decode(memory, beam_width)

    def _greedy_decode(self, memory):
        device = memory.device
        B = memory.shape[0]
        generated_tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=device)
        logits_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(self.max_length - 1):
            if finished.all():
                break
            tgt_emb = self.embedding(generated_tokens) + self.pos_encoding(torch.arange(generated_tokens.size(1), device=device))
            output = self.transformer_decoder(tgt_emb, memory)
            logits = self.output_layer(output[:, -1, :])  # (B, vocab_size)
            logits_list.append(logits.unsqueeze(1))
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            finished = finished | (next_token.squeeze(-1) == self.eos_index)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        all_logits = torch.cat(logits_list, dim=1)
        predicted_tokens = generated_tokens
        mask = (predicted_tokens == self.eos_index).cumsum(dim=1) == 0
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1)
        lengths = mask.sum(dim=1)
        predicted_tokens = [predicted_tokens[i, :lengths[i]] for i in range(B)]
        all_logits = [all_logits[i, :lengths[i]-1] for i in range(B)]
        return all_logits, predicted_tokens

    def _beam_search_decode(self, memory, beam_width):
        device = memory.device
        B = memory.shape[0]
        
        # Инициализация: для каждого элемента батча создаем начальные beams
        generated_tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=device)
        tgt_emb = self.embedding(generated_tokens) + self.pos_encoding(torch.arange(1, device=device))
        output = self.transformer_decoder(tgt_emb, memory)
        logits = self.output_layer(output[:, -1, :])  # (B, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, vocab_size)
        
        # Берем топ beam_width токенов для каждого батча
        top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)  # (B, beam_width)
        beams = []
        for b in range(B):
            beams.append([(torch.tensor([[self.sos_index, top_tokens[b, k].item()]], dtype=torch.long, device=device), 
                           top_log_probs[b, k].item()) for k in range(beam_width)])

        # Beam search цикл
        for t in range(1, self.max_length - 1):
            new_beams = [[] for _ in range(B)]
            all_finished = True
            
            for b in range(B):
                if not beams[b]:  # Если все beams для этого батча завершены
                    continue
                    
                for tokens, log_prob in beams[b]:
                    if tokens[0, -1] == self.eos_index:
                        new_beams[b].append((tokens, log_prob))
                        continue
                    
                    all_finished = False
                    tgt_emb = self.embedding(tokens) + self.pos_encoding(torch.arange(tokens.size(1), device=device))
                    output = self.transformer_decoder(tgt_emb, memory[b:b+1])  # (1, seq_len, embed_dim)
                    logits = self.output_layer(output[:, -1, :])  # (1, vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)  # (1, vocab_size)
                    top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)  # (1, beam_width)
                    
                    for k in range(beam_width):
                        new_tokens = torch.cat([tokens, top_tokens[:, k].unsqueeze(0)], dim=1)
                        new_log_prob = log_prob + top_log_probs[0, k].item()
                        new_beams[b].append((new_tokens, new_log_prob))
                
                # Сортируем и обрезаем до beam_width
                new_beams[b] = sorted(new_beams[b], key=lambda x: x[1], reverse=True)[:beam_width]
                beams[b] = new_beams[b]
            
            if all_finished:
                break

        # Собираем лучшие последовательности
        predicted_tokens = []
        for b in range(B):
            if beams[b]:
                best_tokens, _ = max(beams[b], key=lambda x: x[1])
                predicted_tokens.append(best_tokens.squeeze(0))
            else:
                predicted_tokens.append(torch.tensor([self.sos_index], device=device))  # Fallback

        return None, predicted_tokens  # Логиты не возвращаем, так как они не нужны для beam search