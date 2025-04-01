import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderAttention(nn.Module):
    """
    Аналог create_decoder_attn. В Lua-коде:
      et = W_h * h_{t-1} + W_v * v + ... => softmax => контекст
    Упрощённо используем схему "Luong attention" или "Bahdanau".
    """
    def __init__(self, hidden_size):
        super(DecoderAttention, self).__init__()
        # Приблизимся к "Luong general"
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_state, context):
        """
        decoder_state: (batch, hidden_dim)
        context: (batch, source_len, hidden_dim)  -- это H'*W' позиций
        Возвращаем:
         - контекстный вектор (batch, hidden_dim)
         - веса внимания (batch, source_len)
        """
        # 1) матриц-множим context на decoder_state
        # 2) softmax
        # 3) контекст = sum( alpha_i * context_i )

        # (batch, hidden_dim) -> (batch, 1, hidden_dim)
        dec_state_exp = decoder_state.unsqueeze(1)
        # W * dec_state
        dec_proj = self.W(dec_state_exp)  # (batch,1,hidden_dim)

        # Скалярное произведение с each context
        # context shape: (batch, source_len, hidden_dim)
        # -> score = context * dec_proj^T
        scores = torch.bmm(context, dec_proj.transpose(1,2))  # (batch, source_len, 1)
        scores = scores.squeeze(-1)  # (batch, source_len)

        alpha = F.softmax(scores, dim=1)  # (batch, source_len)
        alpha_3d = alpha.unsqueeze(1)     # (batch, 1, source_len)

        # Теперь контекст = alpha_3d * context
        attended = torch.bmm(alpha_3d, context)  # (batch, 1, hidden_dim)
        attended = attended.squeeze(1)           # (batch, hidden_dim)

        return attended, alpha
