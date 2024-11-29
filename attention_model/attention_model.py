"""Transformer block and full model definitions"""

from typing import Optional, Tuple
import torch
from attention_model.attention_layers import (
    GroupedQueryAttention,
    RotaryPositionalEncoding,
    SwiGLUFeedForward,
)


class TransformerBlock(torch.nn.Module):
    """Transformer block with layer norm."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_query: int,
        rope: RotaryPositionalEncoding,
        n_ff_expansion: int = 4,
        dropout_pct: float = 0.4,
        swish_beta: float = 1.0,
    ) -> None:
        """
        Args:
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature length for each query is (`d_model / n_head / n_query`).
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head (int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
            dropout_pct (float): Dropout probability applied to each element in the attention
                probabilities before matmul with the Value matrix in SDPA function.
            rope (torch.nn.Module): Rotary Positional Encoding instance to use. Allows for
                sharing the sin/cos cache between layers.
            n_ff_expansion (int): Multiplicative factor for expanding the feature dimension when
                passing through the feedforward layers.
            swish_beta (float): Beta value to apply in the SwiGLU feedforward layer.
        """
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=d_model)

        self.attention = GroupedQueryAttention(
            d_model=d_model, n_head=n_head, n_query=n_query, dropout_pct=dropout_pct, rope=rope
        )

        expanded_dim: int = d_model * n_ff_expansion
        self.feedforward = SwiGLUFeedForward(
            n_feat=d_model, n_feat_exp=expanded_dim, swish_beta=swish_beta
        )

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for transformer block.

        Args:
            x (torch.Tensor): Input embedding (batch_size, seq_len, d_model)
            y (torch.Tensor): Optional batched tensor of shape (n_b, n_seq, d_model), if present,
                the layer calculates cross-attention between x and y by using x to compute the query
                and y to compute the keys and values.
            mask (torch.Tensor): Optional mask to apply for causal attention.

        Returns:
            output (torch.Tensor):
        """
        x = x + self.attention.forward(
            self.norm1(x), self.norm1(y) if y is not None else y, mask=mask
        )
        return x + self.feedforward.forward(self.norm2(x))


class TransformerModel(torch.nn.Module):
    """Full decoder-only transformer architecture. Accepts input features and lagged responders and
    outputs a discrete probability distribution and predicted 
    
    Architecture:
        input = features, lagged responders
        Create embeddings: input -> linear projection -> embedding
        Cross attention: 
            feature_embedding ->  keys, values
            responder_embedding -> queries
        `n_blocks` self attention blocks
        Predict:
            Distribution over responder variables as discrete PDF with `n_bins` bins.
            Offset from each bin center to the predicted value in each bin
    """

    def __init__(
        self,
        n_blocks: int,
        n_feature_len: int,
        n_responder_len: int,
        n_output_bins: int,
        d_model: int,
        n_head: int,
        n_query: int,
        n_ff_expansion: int = 4,
        dropout_pct: float = 0.4,
        swish_beta: float = 1.0,
    ) -> None:
        """
        Args:
            n_blocks (int): Number of transformer blocks to create.
            n_feature_len (int): Length for of the input features.
            n_responder_len (int): Length of the input lagged responders.
            n_output_bins (int): Number of bins to predict in the output discrete probability
                distribution.
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature length for each query is (`d_model / n_head / n_query`).
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head (int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
            dropout_pct (float): Dropout probability applied to each element in the attention
                probabilities before matmul with the Value matrix in SDPA function.
            rope (torch.nn.Module): Rotary Positional Encoding instance to use. Allows for
                sharing the sin/cos cache between layers.
            n_ff_expansion (int): Multiplicative factor for expanding the feature dimension when
                passing through the feedforward layers.
            swish_beta (float): Beta value to apply in the SwiGLU feedforward layer.
        """
        super().__init__()

        self.n_feature_len = n_feature_len
        self.n_responder_len = n_responder_len

        # Embeddings are just linear layers since the inputs are real-valued.
        self.feature_embedding = torch.nn.Linear(in_features=n_feature_len, out_features=d_model)
        self.responder_embedding = torch.nn.Linear(
            in_features=n_responder_len, out_features=d_model
        )

        self.rope = RotaryPositionalEncoding(d_model=d_model)
        self.out_norm = torch.nn.LayerNorm(d_model)
        self.logit_linear = torch.nn.Linear(in_features=d_model, out_features=n_output_bins)
        self.offset_linear = torch.nn.Linear(in_features=d_model, out_features=n_output_bins)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.cross_attn_layer = TransformerBlock(
            n_head=n_head,
            n_query=n_query,
            d_model=d_model,
            rope=self.rope,
            n_ff_expansion=n_ff_expansion,
            swish_beta=swish_beta,
            dropout_pct=dropout_pct,
        )

        self.layers = [
            TransformerBlock(
                n_head=n_head,
                n_query=n_query,
                d_model=d_model,
                rope=self.rope,
                n_ff_expansion=n_ff_expansion,
                swish_beta=swish_beta,
                dropout_pct=dropout_pct,
            )
            for _ in range(n_blocks)
        ]

        self.cross_attn_mask: Optional[torch.Tensor] = None
        self.self_attn_mask: Optional[torch.Tensor] = None

    def create_causal_mask(self, seq_len: int, device: torch.DeviceObjType) -> torch.Tensor:
        """Get the self and cross attention masks for the input sequence length.

        Args:
            seq_len (int) Length of the sequence to generate a mask for.
            device (torch.DeviceObjType): Pytorch device object to allocate the masks on.

        Returns:
            mask (torch.Tensor): Bool tensor with upper triangular set to True, including diagonal.
        """
        return torch.triu(torch.ones(seq_len, seq_len, dtype=bool, device=device), diagonal=1)

    def forward(self, features: torch.Tensor, responders: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. Treat responder regression as a classification problem to
        obtain an empirical distribution over the possible responder values, and then regress an
        offset from each bin center to the predicted value for each bin.

        Args:
            features (torch.Tensor): Input features (batch_size, seq_len, n_feature_len)
            responders (torch.Tensor): Input lagged responders (batch_size, seq_len, n_responder_len)
                where the feature vector `t` and responder vector `t-1` are paired.

        Returns:
            responder_distribution (torch.Tensor): (batch_size, seq_len, n_output_bins) probability
                distribution over binned values.
            responder_offset (torch.Tensor): (batch_size, seq_len, n_output_bins) predicted offset
                from the center of each bin to the predicted value within that bin.
        """
        batch_size, seq_len, n_feature_len = features.shape
        n_responder_len = responders.shape[-1]

        assert responders.shape[0] == batch_size
        assert responders.shape[1] == seq_len
        assert n_responder_len == self.n_responder_len
        assert n_feature_len == self.n_feature_len

        feature_emb = self.feature_embedding(features)
        responder_emb = self.responder_embedding(responders)

        causal_mask = self.create_causal_mask(seq_len=seq_len, device=features.device)
        out = self.cross_attn_layer.forward(x=responder_emb, y=feature_emb, mask=causal_mask)

        for layer in self.layers:
            out = layer.forward(x=out, mask=causal_mask)

        out_norm = self.out_norm(out)
        responder_distribution = self.softmax(self.logit_linear(out_norm))
        responder_offset = self.offset_linear(out_norm)

        return responder_distribution, responder_offset
