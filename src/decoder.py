import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size, embed_dim=2176, num_heads=16, num_layers=12, ffn_dim=8192, max_length=300, sos_index=1, eos_index=2):
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

    def forward(self, tgt_tokens, memory):
        device = memory.device
        if tgt_tokens is not None:
            # Режим обучения: полная последовательность предоставлена
            B, T = tgt_tokens.shape
            tgt_emb = self.embedding(tgt_tokens) + self.pos_encoding(torch.arange(T, device=device))
            output = self.transformer_decoder(tgt_emb, memory)
            logits = self.output_layer(output)  # (B, T, vocab_size)
            return logits
        else:
            # Режим инференса: генерация последовательности с нуля
            B = memory.shape[0]
            # Инициализация с SOS токеном
            generated_tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=device)
            logits_list = []
            finished = torch.zeros(B, dtype=torch.bool, device=device)  # Флаги завершения для каждого батча

            for t in range(self.max_length):
                if finished.all():  # Если все последовательности завершены, выходим
                    break
                # Эмбеддинги для текущей последовательности
                tgt_emb = self.embedding(generated_tokens) + self.pos_encoding(torch.arange(generated_tokens.size(1), device=device))
                # Декодируем
                output = self.transformer_decoder(tgt_emb, memory)
                # Логиты для последнего токена
                logits = self.output_layer(output[:, -1, :])  # (B, vocab_size)
                logits_list.append(logits.unsqueeze(1))  # Сохраняем логиты
                
                # Предсказываем следующий токен
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                # Обновляем флаги завершения
                finished = finished | (next_token.squeeze(-1) == self.eos_index)
                # Добавляем новый токен в последовательность
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Собираем логиты и обрезаем последовательности по eos_index
            all_logits = torch.cat(logits_list, dim=1)  # (B, generated_length, vocab_size)
            predicted_tokens = generated_tokens  # (B, generated_length+1), включая SOS
            
            # Обрезаем по eos_index
            max_len = predicted_tokens.size(1)
            mask = (predicted_tokens == self.eos_index).cumsum(dim=1) == 0  # True до первого eos
            mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1)  # Сохраняем SOS
            lengths = mask.sum(dim=1)  # Длина каждой последовательности до eos включительно
            
            # Применяем маску к токенам
            predicted_tokens = [predicted_tokens[i, :lengths[i]] for i in range(B)]
            all_logits = [all_logits[i, :lengths[i]-1] for i in range(B)]  # Логиты до последнего предсказания
            
            return all_logits, predicted_tokens
