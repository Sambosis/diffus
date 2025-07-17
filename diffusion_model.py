import torch
import torch.nn as nn
import math
import config

class PositionalEncoding(nn.Module):
    """
    Standard Transformer Positional Encoding.

    Injects information about the relative or absolute position of tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings so that they can be summed. This implementation uses
    sine and cosine functions of different frequencies. It is adapted for
    `batch_first=True` Tensors.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimension of the model (and embeddings).
            dropout (float): The dropout rate.
            max_len (int): The maximum possible length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Create a positional encoding matrix of shape [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        #  Register 'pe' as a buffer, so it's part of the model's state but not a parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
                              Shape: [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: The input tensor with positional encodings added.
                          Shape: [batch_size, seq_len, d_model].
        """
        # Add the positional encodings up to the sequence length of the input.
        # self.pe is [1, max_len, d_model], we select [: , :x.size(1), :] to match seq_len.
        # Broadcasting handles the batch dimension.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes diffusion timestep `t` into a vector of size `dim` using sinusoidal functions.
    This allows the model to be conditioned on the diffusion timestep.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim (int): The dimension of the embedding vector.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time (torch.Tensor): A tensor of shape (batch_size,) representing the timesteps.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, dim) representing timestep embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]  # (batch_size, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (batch_size, dim)
        
        # Handle the case where dim is odd
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


class TextDiffusionModel(nn.Module):
    """
    A Transformer-based model for denoising text embeddings.

    This model takes noisy text embeddings and a diffusion timestep as input,
    and it predicts the original noise that was added to the embeddings.
    The architecture uses a Transformer Encoder to process the sequence and is
    conditioned on the timestep via an MLP projection.
    """

    def __init__(self):
        super().__init__()
        self.d_model = config.EMBED_DIM

        # This embedding layer is part of the model so the trainer can access it
        # for converting token IDs to embeddings before the diffusion process starts,
        # and for converting model output back to token logits during sampling.
        self.token_embedding = nn.Embedding(config.TOKENIZER_VOCAB_SIZE, self.d_model)

        # Positional encoding for the token sequence
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, 
            dropout=config.DROPOUT, 
            max_len=config.MAX_SEQ_LEN
        )

        # Timestep embedding network (MLP)
        time_embed_dim = self.d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.d_model),
            nn.Linear(self.d_model, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, self.d_model),
        )

        # Core Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True  # Ensure input format is [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.NUM_LAYERS
        )

        # Final linear layer to project the transformer output back to the embedding dimension
        self.output_layer = nn.Linear(self.d_model, self.d_model)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the denoising model.

        Args:
            x_noisy (torch.Tensor): The noisy embeddings at a given timestep.
                                    Shape: [batch_size, seq_len, d_model].
            t (torch.Tensor): The current diffusion timestep for each item in the batch.
                              Shape: [batch_size].

        Returns:
            torch.Tensor: The predicted noise. Shape: [batch_size, seq_len, d_model].
        """
        # 1. Add standard positional encodings for token sequence order.
        x_pos_encoded = self.pos_encoder(x_noisy)
        
        # 2. Compute timestep embeddings. This conditions the model on `t`.
        t_emb = self.time_mlp(t)  # Shape: [batch_size, d_model]
        
        # 3. Add timestep embedding to the sequence.
        # Reshape t_emb to [batch_size, 1, d_model] for broadcasting.
        # This adds the same time information to every token in a sequence.
        x_conditioned = x_pos_encoded + t_emb.unsqueeze(1)

        # 4. Pass the conditioned sequence through the Transformer encoder.
        # No source mask is needed for an unconditional denoising model.
        transformer_output = self.transformer_encoder(x_conditioned)

        # 5. Project the output to predict the noise component.
        predicted_noise = self.output_layer(transformer_output)

        return predicted_noise