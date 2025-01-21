import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Аналог a(·) из оригинала: вычисление et и softmax.
    Но чуть упрощён, без отдельного "input_feed" внутри.
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # W_h, W_s и v^T из Bahdanau
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        """
        decoder_state: (batch, hidden_size) – h_t
        encoder_outputs: (batch, seq_len, hidden_size)
                        – набор "аннотаций" из энкодера
        Возвращает:
          context_vector: (batch, hidden_size),
          attn_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()

        # Расширяем decoder_state до (batch, seq_len, hidden_size)
        decoder_state_expanded = decoder_state.unsqueeze(1).expand(-1, seq_len, -1)

        # Вычисляем e_t = v^T * tanh( W_h * hi + W_s * h_t )
        scores = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_state_expanded))
        scores = self.v(scores).squeeze(-1)  # (batch, seq_len)

        # αt = softmax(e_t)
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # контекст = ∑ α_i * hi
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, hidden_size)

        return context, attn_weights
