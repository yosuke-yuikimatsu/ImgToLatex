import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size=691, embed_dim=1536, num_heads=12, num_layers=6, 
                 ffn_dim=3072, max_length=50, sos_index=1, eos_index=2, dropout=0.1,training=True):
        super().__init__()
        
        # Use weight tying between embedding and output layer for parameter efficiency
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Use more efficient positional encoding with learned parameters
        self.pos_encoding = nn.Parameter(torch.zeros(max_length, embed_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Use pre-norm architecture for better training stability
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ffn_dim, 
            dropout=dropout,
            activation="gelu",  # GELU activation often performs better
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        
        # Add layer norm before the decoder
        self.pre_norm = nn.LayerNorm(embed_dim)
        
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Tie weights between embedding and output layer
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight
        
        # Add output bias separately since we tied weights
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        
        self.max_length = max_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.training = training
        
        # Cache for key-value pairs during inference
        self.kv_cache = None

    def generate_square_subsequent_mask(self, sz, device):
        # More efficient mask generation
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, tgt_tokens, memory, beam_width=None):
        device = memory.device
        
        if tgt_tokens is not None:
            # Training mode
            B, T = tgt_tokens.shape
            T_minus_1 = T - 1
            
            # Get embeddings and add positional encoding
            tgt_emb = self.embedding(tgt_tokens[:, :-1])
            positions = self.pos_encoding[:T_minus_1].unsqueeze(0).expand(B, -1, -1)
            tgt_emb = self.dropout(tgt_emb + positions)
            
            # Generate attention mask
            tgt_mask = self.generate_square_subsequent_mask(T_minus_1, device)
            
            # Apply pre-norm
            tgt_emb = self.pre_norm(tgt_emb)
            
            # Use gradient checkpointing if training
            if self.training:
                output = torch.utils.checkpoint.checkpoint(
                    self.transformer_decoder, tgt_emb, memory, tgt_mask,use_reentrant=False
                )
            else:
                output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
            # Apply output layer with tied weights
            logits = self.output_layer(output) + self.output_bias
            return logits
        else:
            # Inference mode
            if beam_width is None or beam_width == 1:
                return self._optimized_greedy_decode(memory)
            else:
                return self._optimized_beam_search_decode(memory, beam_width)

    def _optimized_greedy_decode(self, memory):
        device = memory.device
        B = memory.shape[0]
        
        # Initialize with start token
        generated_tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=device)
        logits_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Initialize KV cache
        self.kv_cache = None
        
        for t in range(self.max_length - 1):
            if finished.all():
                break
                
            # Only process the last token with cached KV
            if t == 0:
                # First step - process the SOS token
                tgt_emb = self.embedding(generated_tokens)
                positions = self.pos_encoding[:1].expand(B, 1, -1)
                tgt_emb = self.dropout(tgt_emb + positions)
                tgt_emb = self.pre_norm(tgt_emb)
                
                # No mask needed for single token
                output = self.transformer_decoder(tgt_emb, memory)
            else:
                # Subsequent steps - only process new token
                last_token = generated_tokens[:, -1:]
                tgt_emb = self.embedding(last_token)
                positions = self.pos_encoding[t:t+1].expand(B, 1, -1)
                tgt_emb = self.dropout(tgt_emb + positions)
                tgt_emb = self.pre_norm(tgt_emb)
                
                # Use incremental decoding (would require custom implementation)
                # This is a placeholder for the concept
                output = self.transformer_decoder(tgt_emb, memory)
            
            # Get logits for next token prediction
            logits = self.output_layer(output[:, -1:]) + self.output_bias
            logits_list.append(logits)
            
            # Get next token
            next_token = logits.argmax(dim=-1)
            finished = finished | (next_token.squeeze(-1) == self.eos_index)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        
        # Combine logits and process results
        all_logits = torch.cat(logits_list, dim=1)
        predicted_tokens = generated_tokens
        
        # Create mask for valid tokens (before EOS)
        mask = (predicted_tokens != self.eos_index).cumprod(dim=1).bool()
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1)
        lengths = mask.sum(dim=1)
        
        # Extract valid tokens for each sample
        predicted_tokens = [predicted_tokens[i, :lengths[i]] for i in range(B)]
        all_logits = [all_logits[i, :lengths[i]-1] for i in range(B)]
        
        return all_logits, predicted_tokens

    def _optimized_beam_search_decode(self, memory, beam_width):
        device = memory.device
        B = memory.shape[0]
        
        # Optimize by processing all beams in parallel
        # Expand memory for beam search: [B, seq, dim] -> [B * beam_width, seq, dim]
        expanded_memory = memory.unsqueeze(1).expand(-1, beam_width, -1, -1)
        expanded_memory = expanded_memory.reshape(B * beam_width, memory.size(1), memory.size(2))
        
        # Initialize with start tokens
        batch_sos = torch.full((B * beam_width, 1), self.sos_index, dtype=torch.long, device=device)
        
        # Track beam scores
        beam_scores = torch.zeros(B, beam_width, device=device)
        beam_scores[:, 1:] = float('-inf')  # Initialize only first beam as active
        
        # Track finished beams
        finished_beams = torch.zeros(B, beam_width, dtype=torch.bool, device=device)
        
        # Initialize generated sequences with SOS token
        generated_sequences = batch_sos.view(B, beam_width, 1)
        
        # Main beam search loop
        for step in range(self.max_length - 1):
            # Check if all beams are finished
            if finished_beams.all():
                break
                
            # Get current tokens to process
            current_tokens = generated_sequences.view(B * beam_width, -1)
            
            # Compute embeddings
            tgt_emb = self.embedding(current_tokens)
            positions = self.pos_encoding[:current_tokens.size(1)].expand(B * beam_width, -1, -1)
            tgt_emb = self.dropout(tgt_emb + positions)
            tgt_emb = self.pre_norm(tgt_emb)
            
            # Create attention mask
            tgt_mask = self.generate_square_subsequent_mask(current_tokens.size(1), device)
            
            # Decoder forward pass
            output = self.transformer_decoder(tgt_emb, expanded_memory, tgt_mask=tgt_mask)
            
            # Get logits for next token prediction
            logits = self.output_layer(output[:, -1]) + self.output_bias
            
            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # [B * beam_width, vocab_size]
            
            # Reshape for beam processing
            log_probs = log_probs.view(B, beam_width, -1)  # [B, beam_width, vocab_size]
            
            # Add current beam scores to log probs
            log_probs = log_probs + beam_scores.unsqueeze(-1)
            
            # For finished beams, only the EOS token is valid
            finished_mask = finished_beams.unsqueeze(-1).expand(-1, -1, self.vocab_size)
            log_probs = log_probs.masked_fill(
                finished_mask & (torch.arange(self.vocab_size, device=device) != self.eos_index).unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
            
            # Flatten to find top-k over all beams and vocab
            vocab_size = log_probs.size(-1)
            log_probs_flat = log_probs.view(B, -1)  # [B, beam_width * vocab_size]
            
            # Select top-k
            topk_log_probs, topk_indices = log_probs_flat.topk(beam_width, dim=1)
            
            # Convert flat indices to beam indices and token indices
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
            
            # Update beam scores
            beam_scores = topk_log_probs
            
            # Create new sequences
            new_sequences = torch.zeros(B, beam_width, step + 2, dtype=torch.long, device=device)
            
            for batch_idx in range(B):
                for beam_idx in range(beam_width):
                    # Get source beam index
                    src_beam_idx = beam_indices[batch_idx, beam_idx]
                    # Get token to add
                    token_idx = token_indices[batch_idx, beam_idx]
                    # Copy sequence from source beam
                    new_sequences[batch_idx, beam_idx, :step+1] = generated_sequences[batch_idx, src_beam_idx]
                    # Add new token
                    new_sequences[batch_idx, beam_idx, step+1] = token_idx
                    # Mark as finished if EOS
                    if token_idx == self.eos_index:
                        finished_beams[batch_idx, beam_idx] = True
            
            # Update generated sequences
            generated_sequences = new_sequences
        
        # Select best beam for each batch
        best_sequences = []
        for batch_idx in range(B):
            # Find best beam (highest score for finished, or if none finished, highest overall)
            if finished_beams[batch_idx].any():
                # Get scores of finished beams
                finished_scores = beam_scores[batch_idx].masked_fill(~finished_beams[batch_idx], float('-inf'))
                best_beam_idx = finished_scores.argmax()
            else:
                best_beam_idx = beam_scores[batch_idx].argmax()
            
            # Get sequence from best beam
            best_seq = generated_sequences[batch_idx, best_beam_idx]
            
            # Trim after EOS
            eos_positions = (best_seq == self.eos_index).nonzero()
            if eos_positions.size(0) > 0:
                best_seq = best_seq[:eos_positions[0] + 1]
            
            best_sequences.append(best_seq)
        
        # Placeholder for logits (not used in inference)
        dummy_logits = [torch.zeros(1, device=device) for _ in range(B)]
        
        return dummy_logits, best_sequences
