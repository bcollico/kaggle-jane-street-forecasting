"""Transformer block and full model definitions"""

from typing import Optional
from copy import deepcopy
import torch
from attention_layers.attention_layers import (
    InfiniGroupedQueryAttention,
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
        self.feature_embedding = torch.nn.Linear(
            in_features=n_feature_len, out_features=d_model, bias=True
        )
        self.responder_embedding = torch.nn.Linear(
            in_features=n_responder_len, out_features=d_model, bias=True
        )

        # Create embeddings for all possible uint8 symbols, most of these will not be updated, but
        # to make the model extensible to new symbols let's allocate enough for any 8-bit int ID.
        # Start embeddings at zero so that unseen embeddings have no contribution to the features.
        self.symbol_embedding_f = self.create_embedding(256, d_model)
        self.symbol_embedding_r = self.create_embedding(256, d_model)

        # Date embedding. Assume that Date IDs are consistent such that when we take modulus with
        # 365 we get the same day of the year (potentially not correct due to leap year).
        self.date_embedding_f = self.create_embedding(365, d_model)
        self.date_embedding_r = self.create_embedding(365, d_model)

        # Absolute time embedding. The training dataset has <1000 absolute time indices. It's
        # uncertain how I should be using these since the actual time between time_ids is not fixed,
        # but we should use some form of absolute time embedding to capture intra-day trends.
        self.time_embedding_f = self.create_embedding(1000, d_model)
        self.time_embedding_r = self.create_embedding(1000, d_model)

        # Positional encoding var.
        self.rope = RotaryPositionalEncoding(d_model=d_model, enable_sin_cos_caching=True)

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

    @staticmethod
    def create_embedding(dim_in: int, dim_out: int) -> torch.nn.Embedding:
        """Create an embedding layer initialized with weights sampled from N(0, 0.1)."""
        return torch.nn.Embedding.from_pretrained(
            torch.normal(mean=0.0, std=0.1, size=(dim_in, dim_out)),
            freeze=False,
            scale_grad_by_freq=False,
        )

    def reset_memory(self) -> None:
        """Reset the memory matrices for each transformer block layer."""
        if isinstance(self.cross_attn_layer.attention, InfiniGroupedQueryAttention):
            self.cross_attn_layer.attention.reset_memory()

        layer: TransformerBlock
        for layer in self.layers:
            if isinstance(layer.attention, InfiniGroupedQueryAttention):
                layer.attention.reset_memory()

    @staticmethod
    def create_causal_mask(id_mat_1: torch.Tensor, id_mat_2: torch.Tensor) -> torch.Tensor:
        """Get the causal attention mask for the input sequence lengths. return a mask that
        allows for attention between tokens with the same ID value or a lesser ID value.

        E.g. id_mat_1 = [1, 1, 2, 2, 2, 3], id_mask_2 = [0, 1, 2, 2, 3]

        causal attention mask -- attend to rows with the same ID or less.

        output = [False, False,  True,  True,  True],
                 [False, False,  True,  True,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False, False]]

        Args:
            id_mat_1 (torch.Tensor): Mask of ID values to use to create the mask. Items with the same
                ID can attend to each other. Items can attend to values with smaller ID values.
                (..., seq_len_1)
            id_mat_2 (torch.Tensor): Mask of ID values to compare the first mask against.

        Returns:
            causal_mask (torch.Tensor): Bool tensor id_mat_1.T < id_mat_2, (1, seq_len_1, seq_len_2)
        """
        return id_mat_1.unsqueeze(1).transpose(-1, -2) < id_mat_2.unsqueeze(1)

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
            features (torch.Tensor): Input features (batch_size, features_seq_len, n_feature_len)
                from dt_1 ... dt_n
            responders (torch.Tensor): Input lagged responders from dt_0 .. dt_n-1.
                (batch_size, responder_seq_len, n_responder_len) where the feature vector `t` and
                responder vector `t` are paired. The responders are masked in the forward pass so
                that the model doesn't get any "future" information in the attention scores.

        Returns:
            predicted_responders (torch.Tensor): (batch_size, seq_len, n_responder_len) responder
                predictions.
        """

        # The responders and the features are not necessarily the same length since the responders
        # are lagged by one date/time.
        feature_seq_len = features.shape[-2]
        responder_seq_len = responders.shape[-2]

        # Symbol and time embeddings use their absolute values.
        # Date embedding uses value modulo 365 to hopefully capture long-term cyclical trends.
        # Summed input embeddings. Consider trying to concat or append.
        f_embedding = (
            self.time_embedding_f(time_ids[..., -feature_seq_len:])
            + self.date_embedding_f(date_ids[..., -feature_seq_len:] % 365)
            + self.symbol_embedding_f(symbol_ids[..., -feature_seq_len:])
        )
        r_embedding = (
            self.time_embedding_r(time_ids[..., :responder_seq_len])
            + self.date_embedding_r(date_ids[..., :responder_seq_len] % 365)
            + self.symbol_embedding_r(symbol_ids[..., :responder_seq_len])
        )

        # Create feature and responder embeddings. Index into the embedding to get the date/time
        # information relevant for features and responders. The dates/times/symbols are provided
        # in the same order (chronological + sequential by sequence ID) so that the
        feature_emb = self.feature_embedding(features) + f_embedding
        responder_emb = self.responder_embedding(responders) + r_embedding

        # Create the masks for cross and self attention.
        with torch.no_grad():
            # Cross attention mask is created by comparing the feature time IDs against the lagged
            # time IDs (advanced by one index since we want the current t to attend to lags < t).
            cross_attn_mask: torch.Tensor = self.create_causal_mask(
                id_mat_1=time_ids[..., -feature_seq_len:],
                id_mat_2=time_ids[..., :responder_seq_len] + 1,
            ).unsqueeze(1)

        # Cross attention between the features and their lagged responders.
        out = self.cross_attn_layer.forward(x=feature_emb, y=responder_emb, mask=cross_attn_mask)

        with torch.no_grad():
            # The self attention mask compares the feature Time IDs to themselves.
            self_attn_mask: torch.Tensor = self.create_causal_mask(
                id_mat_1=time_ids[..., -feature_seq_len:], id_mat_2=time_ids[..., -feature_seq_len:]
            ).unsqueeze(1)

        # Self attention layers.
        for layer in self.layers:
            out = layer.forward(x=out, mask=self_attn_mask)

        out_norm = self.out_norm(out)

        # Predict the responder variables.
        predictions = self.logit_linear(out_norm)

        # Clamp the predictions to [-5, 5].
        torch.clamp(predictions, min=-5.0, max=5.0)

        return predictions
