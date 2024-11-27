"""Transformer block and full model definitions"""

from typing import Optional
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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for transformer block.

        Args:
            x (torch.Tensor): Input embedding (batch_size, seq_len, d_model)
            y (torch.Tensor): Optional batched tensor of shape (n_b, n_seq, d_model), if present,
                the layer calculates cross-attention between x and y by using x to compute the query
                and y to compute the keys and values.

        Returns:
            output (torch.Tensor):
        """
        x = x + self.attention(self.norm1(x), self.norm1(y) if y else y)
        return x + self.feedforward(self.norm2(x))


class TransformerModel(torch.nn.Module):

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

    def forward(self, features: torch.Tensor, responders: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. Treat responder regression as a classification problem to
        obtain an empirical distribution over the possible responder values, and then regress an
        offset from each bin center to the predicted value for each bin.

        Args:
            features (torch.Tensor): Input features (batch_size, seq_len, n_feature_len)
            responders (torch.Tensor): Input lagged responders (batch_size, seq_len, n_responder_len).

        Returns:
            responder_distribution (torch.Tensor): (batch_size, seq_len, n_output_bins) probability
                distribution over binned values.
            responder_offset (torch.Tensor): (batch_size, seq_len, n_output_bins) predicted offset
                from the center of each bin to the predicted value within that bin.
        """
        batch_size, seq_len, n_feature_len = features.shape
        _, _, n_responder_len = features.shape

        assert responders.shape[0] == batch_size
        assert responders.shape[1] == seq_len
        assert n_responder_len == self.n_responder_len
        assert n_feature_len == self.n_feature_len

        feature_emb = self.feature_embedding(features)
        responder_emb = self.responder_embedding(responders)

        out = self.cross_attn_layer.forward(x=responder_emb, y=feature_emb)

        for layer in self.layers:
            out = layer(out)

        out_norm = self.out_norm(out)
        responder_distribution = self.softmax(self.logit_linear(out_norm))
        responder_offset = self.offset_linear(out_norm)

        return responder_distribution, responder_offset

