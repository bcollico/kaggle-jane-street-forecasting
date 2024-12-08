"""Transformer block and full model definitions"""

from typing import Optional, Tuple
from copy import deepcopy
import torch
from attention_model.attention_layers import (
    InfiniGroupedQueryAttention,
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

        self.attention = InfiniGroupedQueryAttention(
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
        if y is not None:
            x = x + self.attention.forward(self.norm1(x), self.norm1(y), mask=mask)
        else:
            x = x + self.attention.forward(self.norm1(x), None, mask=mask)
        return x + self.feedforward.forward(self.norm2(x))


class TransformerModel(torch.nn.Module):
    """Full decoder-only transformer architecture. Accepts input features and lagged responders and
    outputs a discrete probability distribution and predicted

    Architecture:
        input = features, lagged responders
        Create embeddings: input -> linear projection -> embedding
        Cross attention:
            feature_embedding ->  key projection, value projection -> keys, values
            responder_embedding -> query projection -> queries
        `n_blocks` self attention blocks
        Predict:
            Value of the responders.
    """

    def __init__(
        self,
        n_blocks: int,
        n_feature_len: int,
        n_responder_len: int,
        d_model: int,
        n_head: int,
        n_query: int,
        n_ff_expansion: int = 4,
        dropout_pct: float = 0.3,
        swish_beta: float = 1.0,
    ) -> None:
        """
        Args:
            n_blocks (int): Number of transformer blocks to create.
            n_feature_len (int): Length for of the input features.
            n_responder_len (int): Length of the input lagged responders.
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

        # Create embeddings for all possible uint8 symbols, most of these will not be updated, but
        # to make the model extensible to new symbols let's allocate for more than we need.
        # Start embeddings at zero so that unseen embeddings have no contribution to the features.
        self.symbol_embedding = torch.nn.Embedding.from_pretrained(
            torch.zeros(256, d_model), freeze=False
        )

        # Relative date embedding. Assume that we'll have a small window of potential date ranges
        # for an input sample. E.g. input dim of 8 means that we can only span up to 8 days in an
        # input sequence. We assume that intra-day trends will be captured via this embedding and
        # seasonal trends via a memory mechanism. Perhaps this will have to be expanded to better
        # generalize to sequences across many more days than we have here.
        self.date_embedding = torch.nn.Embedding.from_pretrained(
            torch.zeros(365, d_model), freeze=False
        )

        # Absolute time embedding. The training dataset has <1000 absolute time indices. It's
        # uncertain how I should be using these since the actual time between time_ids is not fixed,
        # but we should use some form of absolute time embedding to capture inter-day trends.
        self.time_embedding = torch.nn.Embedding.from_pretrained(
            torch.zeros(1000, d_model), freeze=False
        )

        # Positional encoding var.
        self.rope = RotaryPositionalEncoding(d_model=d_model)

        self.out_norm = torch.nn.LayerNorm(d_model)
        self.logit_linear = torch.nn.Linear(in_features=d_model, out_features=n_responder_len)
        self.softmax = torch.nn.Softmax(dim=-1)

        # TODO rethink this cross attention situation -- if the responder at
        # t is paired with the feature at t+1, we could just so regular self-attention all the way
        # through. Could also see if there is benefit to interleaving cross attention with the
        # lagged responders and features throughout
        self.cross_attn_layer = TransformerBlock(
            n_head=n_head,
            n_query=n_query,
            d_model=d_model,
            rope=self.rope,
            n_ff_expansion=n_ff_expansion,
            swish_beta=swish_beta,
            dropout_pct=dropout_pct,
        )

        self.layers = torch.nn.ModuleList(
            [
                deepcopy(
                    TransformerBlock(
                        n_head=n_head,
                        n_query=n_query,
                        d_model=d_model,
                        rope=self.rope,
                        n_ff_expansion=n_ff_expansion,
                        swish_beta=swish_beta,
                        dropout_pct=dropout_pct,
                    )
                )
                for _ in range(n_blocks)
            ]
        )

    def reset_memory(self):
        self.cross_attn_layer.attention.reset_memory()
        for layer in self.layers:
            layer.attention.reset_memory()

    @staticmethod
    def create_causal_masks(id_mat: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the self and cross attention masks for the input sequence length. return masks that
        do not allow for attention from a token to a token in the same row position as well as
        a mask that does.

        E.g. id_mat = [1, 1, 2, 2, 2, 3]

        cross attention mask (don't attend to rows with same ID.)

        output =    [1, 1, 1, 1, 1, 1]
                    [1, 1, 1, 1, 1, 1]
                    [0, 0, 1, 1, 1, 1]
                    [0, 0, 1, 1, 1, 1]
                    [0, 0, 1, 1, 1, 1]
                    [0, 0, 0, 0, 0, 1]

        self attention mask (attend to rows with same ID or lesser IDs)

        output =    [0, 0, 1, 1, 1, 1]
                    [0, 0, 1, 1, 1, 1]
                    [0, 0, 0, 0, 0, 1]
                    [0, 0, 0, 0, 0, 1]
                    [0, 0, 0, 0, 0, 1]
                    [0, 0, 0, 0, 0, 0]

        Args:
            id_mat (torch.Tensor): Mask of ID values to use to create the mask. Items with the same
                ID can attend to each other. Items can attend to values with smaller ID values.
                (..., seq_len, 1)
            diag (bool): Flag whether to allow attention between a query and value at the same row
                index. Set to false for attention between features and responders at the same time
                to avoid data leakage.

        Returns:
            self_attn_mask (torch.Tensor): Bool tensor with upper triangular set to True, including
                diagonal. (..., seq_len, seq_len)
            cross_attn_mask (torch.Tensor): Bool tensor with upper triangular set to True, excluding
                diagonal. (..., seq_len, seq_len)
        """
        mat: torch.Tensor = id_mat == id_mat.transpose(-1, -2)
        cumsum: torch.Tensor = torch.cumsum(mat, dim=-1)
        return (
            (cumsum + torch.logical_not(mat) > mat.sum(dim=-2, keepdim=True)).unsqueeze(0),
            cumsum.bool().unsqueeze(0),
        )

    def create_date_and_time_masks(
        self, time_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the causal attention masks taking into account the date and time IDs.

        Args:
            time_id (torch.Tensor): Time IDs (..., seq_len).
            date_id (torch.Tensor): Date IDs (..., seq_len).

        Returns:
            self_attn_mask (torch.Tensor): Bool tensor with upper triangular set to True, including
                diagonal. (..., seq_len, seq_len)
            cross_attn_mask (torch.Tensor): Bool tensor with upper triangular set to True, excluding
                diagonal. (..., seq_len, seq_len)
        """
        sa_mask_t, ca_mask_t = self.create_causal_masks(id_mat=time_id)
        # TODO figure out if/how to mask on date + time and not just time. Assuming currently that
        # it might be sufficient to mask on time alone when the window is small enough since the
        # time id will wrap around from a large number to a small number when the sequence is
        # split across days.
        return sa_mask_t, ca_mask_t

    def forward(
        self,
        date_ids: torch.Tensor,
        time_ids: torch.Tensor,
        symbol_ids: torch.Tensor,
        features: torch.Tensor,
        responders: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model. Treat responder regression as a classification problem to
        obtain an empirical distribution over the possible responder values, and then regress an
        offset from each bin center to the predicted value for each bin.

        Args:
            date_ids (torch.Tensor): Integer values (batch_size, seq_len, 1) representing the
                absolute date. Used for retrieving a relative time embedding with the time_id.
            time_ids (torch.Tensor): Integer values batch_size, seq_len, 1)  representing the time
                within a date. Used for retrieving a relative time embedding in combination with
                the date_id.
            symbol_ids (torch.Tensor): Integer values (batch_size, seq_len, 1) representing the
                encrypted market ID of the time series. Used to retrieve a security-specific
                embedding.
            features (torch.Tensor): Input features (batch_size, seq_len, n_feature_len)
            responders (torch.Tensor): Input lagged responders (batch_size, seq_len, n_responder_len)
                where the feature vector `t` and responder vector `t` are paired. The responders
                are masked in the forward pass so that the model doesn't get any "future"
                information in the attention scores.

        Returns:
            predicted_responders (torch.Tensor): (batch_size, seq_len, n_responder_len) responder
                predictions.
        """
        n_feature_len = features.shape[-1]
        n_responder_len = responders.shape[-1]

        assert n_responder_len == self.n_responder_len
        assert n_feature_len == self.n_feature_len

        # Symbol and time embeddings use their absolute values.
        symbol_emb = self.symbol_embedding(symbol_ids)
        time_emb = self.time_embedding(time_ids)

        # Date embedding uses a relative value to the start of this sequence since date is
        # unbounded. TODO: try doing date_id % 365 so that it's bounded or (date_id % 365) % 12
        # to get the long-term seasonal embeddings
        date_emb = self.date_embedding(date_ids % 365)

        # Create feature and responder embeddings.
        feature_emb = self.feature_embedding(features)
        responder_emb = self.responder_embedding(responders)

        # Augment feature embedding with time, date, and symbol embeddings.
        feature_emb = feature_emb + time_emb + date_emb + symbol_emb

        # Create the masks for cross and self attention.
        with torch.no_grad():
            self_attn_mask, cross_attn_mask = self.create_date_and_time_masks(time_id=time_ids)

        # Cross attention between the features and their lagged responders.
        out = self.cross_attn_layer.forward(x=responder_emb, y=feature_emb, mask=cross_attn_mask)

        # Self attention layers.
        for layer in self.layers:
            out = layer.forward(x=out, mask=self_attn_mask)

        out_norm = self.out_norm(out)

        # Predict the responder variables.
        predictions = self.logit_linear(out_norm)

        return predictions
