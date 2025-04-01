import torch
import torch.nn as nn
from cnn import CNN
from encoder import TransformerEncoderModule
from decoder import TransformerDecoderModule

class ImageToLatexModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        enc_hidden_dim: int = 1536,  # Reduced from gazillion
        pad_idx: int = 0,
        sos_index: int = 1,
        eos_index: int = 2,
        max_length: int = 50,
        beam_width: int = 5,
        training=True
    ):
        super().__init__()
        self.cnn = CNN(output_channels=enc_hidden_dim)
        self.encoder = TransformerEncoderModule(
            enc_hid_dim=enc_hidden_dim, 
            num_layers=6,  # Reduced from 10
            ffn_dim=3072,  # Reduced from 8192
            num_heads=12,
            training=training   # Increased from 8 for better parallelism
        )
        self.decoder = TransformerDecoderModule(
            vocab_size=vocab_size,
            embed_dim=enc_hidden_dim,
            num_heads=12,  # Increased from 8
            num_layers=6,  # Reduced from 10
            ffn_dim=3072,  # Reduced from 8192
            max_length=max_length,
            sos_index=sos_index,
            eos_index=eos_index,
            training=training
        )
        self.pad_idx = pad_idx
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_length = max_length
        self.beam_width = beam_width
        self.training = training
        
        # Initialize parameters with better defaults
        self._init_parameters()

    def _init_parameters(self):
        # Apply better initialization for faster convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, images, tgt_tokens=None):
        features = self.cnn(images)
        encoder_outputs = self.encoder(features)
        memory = encoder_outputs
        
        if tgt_tokens is not None:
            # Training mode
            logits = self.decoder(tgt_tokens, memory)
            return logits
        else:
            # Inference mode
            logits, predicted_tokens = self.decoder(tgt_tokens, memory, self.beam_width)
            return logits, predicted_tokens
            
    @torch.jit.export
    def generate(self, images):
        """
        Optimized inference method that can be used with TorchScript
        """
        with torch.no_grad():
            features = self.cnn(images)
            memory = self.encoder(features)
            _, predicted_tokens = self.decoder(None, memory, self.beam_width)
            return None , predicted_tokens
