"""Transformer model with Grouped Multi-Head Attention and compressed KV memory.

Memory mechanism based on Infini-attention: https://arxiv.org/pdf/2404.07143
Grouped-Query Attention based on: https://arxiv.org/pdf/2305.13245
"""

from typing import Optional, Tuple
import torch
from torch.utils.checkpoint import checkpoint


class SwiGLUFeedForward(torch.nn.Module):
    """SwiGLU implementation following https://kikaben.com/swiglu-2020/"""

    def __init__(self, n_feat: int, n_feat_exp: int, swish_beta: float = 1.0) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(in_features=n_feat, out_features=n_feat_exp, bias=False)
        self.swiglu_gate = torch.nn.Linear(in_features=n_feat, out_features=n_feat_exp, bias=False)
        self.linear2 = torch.nn.Linear(in_features=n_feat_exp, out_features=n_feat, bias=False)

        self.swish = lambda x, b: x * torch.nn.functional.sigmoid(b * x)
        self.swish_beta = torch.nn.Parameter(data=torch.tensor(swish_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU feed-forward: (swish(xW+b, beta) * (xV+c))U + d

        Args:
            x (torch.Tensor): Input tensor (..., n_feat)

        Returns:
            output (torch.Tensor): Output tensor (..., n_feat)
        """

        def ckpt_fwd(x: torch.Tensor) -> torch.Tensor:
            gate_tensor = self.swiglu_gate(x)
            x = self.swish(self.linear1(x), self.swish_beta)
            return self.linear2(x * gate_tensor)

        return checkpoint(ckpt_fwd, x, use_reentrant=False)


class RotaryPositionalEncoding(torch.nn.Module):
    """Vanilla Rotary Positional Encoding (RoPE) implementation: https://arxiv.org/pdf/2104.09864"""

    def __init__(
        self,
        d_model: int,
        base: int = 10000,
        device: torch.DeviceObjType = torch.device("cuda"),
        enable_sin_cos_caching: bool = False,
    ):
        """
        Args:
            d_model (int): Expected length of the input token embeddings.
            base (int): Base value to raised to the power of -2i/d_model.
            device (torch.DeviceObjType): Device to store sin/cos cache on.
            enable_sin_cos_caching (bool): Flag to enable caching of sin/cos matrices. Increase
                memory use at the expense of runtime.
        """
        super().__init__()

        # d_model needs to be even.
        assert d_model % 2 == 0

        d_model_2: int = int(d_model / 2)
        self.sin_pos: torch.Tensor = torch.empty(0, d_model_2).to(device)
        self.cos_pos: torch.Tensor = torch.empty(0, d_model_2).to(device)

        # theta parameter vector: (1, d_model / 2)
        self.theta: torch.Tensor = torch.tensor(base, device=device) ** (
            -torch.arange(d_model_2, device=device) / (d_model_2)
        ).unsqueeze(0)

        self.use_cache: bool = enable_sin_cos_caching

    def _allocate_sin_cos(self, seq_len: int) -> None:
        """Allocate sin cos caches for rotary positional encoding. Only calculates
        the part of the positional encoding that is not yet stored in the cache. If the input
        len is less than the current cache size, no-ops.

        Args:
            seq_len (int): Length of the sequence to precompute the positional encoding for.
        """
        with torch.no_grad():
            start = self.sin_pos.shape[0]

            if start >= seq_len:
                return

            cos_pos, sin_pos = self._calc_sin_cos_mats(
                start=0, end=seq_len, d_model_2=self.theta.shape[-1], device=self.theta.device
            )

            # Stack the new values in the sin/cos caches.
            self.sin_pos = torch.vstack((self.sin_pos, sin_pos))
            self.cos_pos = torch.vstack((self.cos_pos, cos_pos))

    def _calc_sin_cos_mats(
        self, start: int, end: int, d_model_2: int, device: torch.DeviceObjType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the sin and cos matrices for a given start/end and model dimension."""
        self.theta = self.theta.to(device)
        position: torch.Tensor = torch.arange(start=start, end=end, device=device).unsqueeze(1)
        angles: torch.Tensor = position * self.theta[..., :d_model_2]
        return torch.cos(angles), torch.sin(angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the positional encoding to the input tensor. Updates the
        sin/cos cache if necessary.

        Args:
            x (torch.Tensor): Tensor of shape (..., seq_len, d_model)
        Returns:
            out (torch.Tensor): Tensor of shape (..., seq_len, d_model) with positional encoding.
        """

        # Update the cache if necessary.
        seq_len, d_model = x.shape[-2], x.shape[-1]
        d_model_2: int = int(d_model / 2)

        if self.use_cache:
            self._allocate_sin_cos(seq_len=seq_len)

            self.sin_pos = self.sin_pos.to(x.device)
            self.cos_pos = self.cos_pos.to(x.device)

            cos_pos: torch.Tensor = self.cos_pos[:seq_len, :d_model_2]
            sin_pos: torch.Tensor = self.sin_pos[:seq_len, :d_model_2]
        else:
            cos_pos, sin_pos = self._calc_sin_cos_mats(
                start=0, end=seq_len, d_model_2=d_model_2, device=x.device
            )

        x_even, x_odd = x[..., ::2], x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos_pos - x_odd * sin_pos
        out[..., 1::2] = x_odd * cos_pos + x_even * sin_pos

        return out


class GroupedQueryAttention(torch.nn.Module):
    """Class for a multi-headed grouped attention layer."""

    def __init__(
        self,
        d_model: int,
        n_query: int,
        n_head: int,
        dropout_pct: float = 0.0,
        rope: Optional[RotaryPositionalEncoding] = None,
    ) -> None:
        """
        Initializes the GroupedQueryAttention with a single Key/Value projection and
        multiple query projection matrices.


        Trainable parameters per layer:
            attn projection: d_model * (d_model + 1)
            query projection: d_model**2
            key projection: d_model**2 / n_query
            value projection: d_model**2 / n_query
            memory mixing parameter: 1

            Total = 2 * (1/n_query + 1) * d_model**2 + d_model + 1

        Args:
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature length for each query is (`d_model / n_head / n_query`).
            n_out (int): The size of the output dimension for each projection.
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head (int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
            dropout_pct (float): Dropout probability applied to each element in the attention
                probabilities before matmul with the Value matrix in SDPA function.
            rope (torch.nn.Module): Rotary Positional Encoding instance to use. Allows for
                sharing the sin/cos cache between layers.
        """
        super().__init__()

        # Check that d_model is evenly divisible by the number of heads and number of queries per
        # head.
        assert d_model % n_head == 0
        assert d_model % n_query == 0
        assert d_model % (n_head * n_query) == 0

        self.d_model = d_model
        self.n_query = n_query
        self.n_head = n_head

        self.attn_dropout = torch.nn.Dropout(p=dropout_pct)

        self.positional_encoding = RotaryPositionalEncoding(d_model=d_model) if not rope else rope

        # The feature length per query in each head. Add 0.5 before casting to int to avoid
        # unexpected rounding behavior due to float-precision
        self.d_head = int(d_model / n_query / n_head + 0.5)

        # Projection matrices are sized such that each head is included in a single matmul.

        # d_model x d_model/n_query projections matrices.
        self.key_proj = torch.nn.Linear(
            in_features=d_model, out_features=self.d_head * n_head, bias=False
        )
        self.value_proj = torch.nn.Linear(
            in_features=d_model, out_features=self.d_head * n_head, bias=False
        )

        #  (d_model, d_model) projection matrix
        self.query_proj = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # (d_model, d_model) projection matrix
        self.attn_proj = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # Scaled attention uses the feature length for each query.
        self.attention_scale = 1.0 / torch.sqrt(torch.tensor(self.d_head))

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply SDPA using grouped query attention from https://arxiv.org/pdf/2305.13245.

        Args:
            query (torch.Tensor):: attention query matrix of shape (n_b, n_seq_q, d_model) where
                d_model = n_query * n_head * d_head
            key (torch.Tensor): attention key matrix of shape (n_b, n_seq_kv, d_head * n_head).
            value (torch.Tensor): attention value matrix of shape (n_b, n_seq_kv, d_head * n_head).
            mask (torch.Tensor | None) Optional mask of shape (batch_size, 1, n_seq_q, n_seq_kv) for
                casual attention.

        Returns:
            attention_output (torch.Tensor): Attention output of shape
                (n_b, n_seq_q, n_query * n_head, d_head).
        """
        n_b, n_seq_q = query.shape[:2]
        n_seq_kv = key.shape[-2]

        # (n_b, n_head, d_head, n_seq_q)
        key_t = key.view(n_b, n_seq_kv, self.n_head, self.d_head).permute(0, 2, 3, 1)

        # (n_b, n_head, n_query * n_seq_q, d_head)
        query_t = (
            query.view(n_b, n_seq_q, self.n_head * self.n_query, self.d_head)
            .transpose(1, 2)
            .reshape(n_b, self.n_head, self.n_query * n_seq_q, self.d_head)
        )

        # (n_b, n_head, n_seq_kv, d_head)
        value_t = value.view(n_b, n_seq_kv, self.n_head, self.d_head).transpose(1, 2)

        # (n_b, n_head , n_query * n_seq_q, n_seq_q)
        attention_scores = (query_t @ key_t * self.attention_scale).view(
            n_b, self.n_head * self.n_query, n_seq_q, n_seq_kv
        )

        if mask is not None:
            attention_scores = torch.masked_fill(attention_scores, mask, value=-1e9)
        attention_probabilities = torch.nn.functional.softmax(attention_scores, dim=-1)

        if mask is not None:
            attention_probabilities = torch.masked_fill(attention_probabilities, mask, 0.0)

        # (n_b, n_query * n_head, n_seq_q, d_head) -> (n_b, n_head, n_query * n_seq_q, d_head)
        attention_probabilities = self.attn_dropout(attention_probabilities)
        attention_out = (
            attention_probabilities.view(n_b, self.n_head, self.n_query * n_seq_q, n_seq_kv)
            @ value_t
        )
        return attention_out.view(n_b, self.n_head * self.n_query, n_seq_q, self.d_head).transpose(
            1, 2
        )

    def kqv(
        self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate key query value projections"""
        return self.key_proj(k), self.query_proj(q), self.value_proj(v)

    def calculate_attention_output(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_qkv: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the attention output using scaled dot product attention.

        Args:
            x (torch.Tensor): Input tensor of shape (n_b, n_seq_q, d_model).
            y (torch.Tensor): Optional batched tensor of shape (n_b, n_seq_kv, d_model), if present,
                the layer calculates cross-attention between x and y by using x to compute the query
                and y to compute the keys and values.
            return_qkv (bool): Return the query, key, and value matrices (in that order) in addition
                to the the attention output
            mask (torch.Tensor | None) Optional mask of shape (batch_size, 1, n_seq_q, n_seq_kv) for
                casual attention.
        Output:
            attention_output (torch.Tensor): Attention output of shape
                (n_b, n_seq_q, n_query * n_head, d_head)

        """

        if y is not None:
            key, query, value = self.kqv(q=x, k=y, v=y)
        else:
            key, query, value = self.kqv(q=x, k=x, v=x)

        attention_output = self.scaled_dot_product_attention(
            query=self.positional_encoding.forward(query),
            key=self.positional_encoding.forward(key),
            value=value,
            mask=mask,
        )

        if return_qkv:
            return attention_output, query, key, value

        # (n_b, n_seq_q, n_query * n_head, d_head)
        return attention_output

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Runs the forward pass, computing `n_head` attention heads with `n_query`
        queries per head. If enabled uses the memory mechanism to mix the local attention
        output with the memory attention.

        Args:
            x (torch.Tensor): Batched tensor of shape (n_b, n_seq_q, d_model)
            y (torch.Tensor): Optional batched tensor of shape (n_b, n_seq_kv, d_model), if present,
                the layer calculates cross-attention between x and y by using x to compute the query
                and y to compute the keys and values.
            mask (torch.Tensor | None) Optional mask of shape (batch_size, 1, n_seq_q, n_seq_kv) for
                casual attention.
        Returns:
            (torch.Tensor): Batched tensor of shape (n_b, n_seq_q, d_model)
        """

        n_b, n_seq_q, d_model = x.shape

        # (n_b, n_seq_q, n_query * n_head, d_head)
        attention_out = self.calculate_attention_output(x=x, y=y, mask=mask)

        # (n_b, n_seq_q, n_query * n_head, d_head) -> ((n_b, n_seq_q, d_model))
        return self.attn_proj(attention_out.reshape(n_b, n_seq_q, d_model))


class InfiniGroupedQueryAttention(GroupedQueryAttention):
    """Class for a multi-headed grouped attention layer."""

    def __init__(
        self,
        d_model: int,
        n_query: int,
        n_head: int,
        dropout_pct: float = 0.0,
        rope: Optional[RotaryPositionalEncoding] = None,
    ) -> None:
        """
        Initializes the InfiniGroupedQueryAttention with a single Key/Value projection and
        multiple query projection matrices per head.


        Trainable parameters per layer:
            attn projection: d_model * (d_model + 1)
            query projection: d_model**2
            key projection: d_model**2 / n_query
            value projection: d_model**2 / n_query
            memory mixing parameter: 1

            Total = 2 * (1/n_query + 1) * d_model**2 + d_model + 1

        Args:
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature length for each query is (`d_model / n_head / n_query`).
            n_out (int): The size of the output dimension for each projection.
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head (int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
            dropout_pct (float): Dropout probability applied to each element in the attention
                probabilities before matmul with the Value matrix in SDPA function.
            rope (torch.nn.Module): Rotary Positional Encoding instance to use. Allows for
                sharing the sin/cos cache between layers.
        """
        super().__init__(
            d_model=d_model, n_query=n_query, n_head=n_head, dropout_pct=dropout_pct, rope=rope
        )

        self.mem_activation = lambda x: torch.nn.functional.elu(x) + 1.0

        # Memory parameters. Keep a feature-length memory matrix and normalization vector for each
        # head.
        self.memory: torch.Tensor
        self.register_buffer(
            name="memory",
            tensor=torch.zeros((self.d_head * n_head, self.d_head * n_head)),
            persistent=True,
        )
        self.memory_norm: torch.Tensor
        self.register_buffer(
            name="memory_norm",
            tensor=torch.ones((self.d_head * n_head, 1)),
            persistent=True,
        )

        # Scalar parameter for mixing the attention scores from the current forward pass with the
        # memory context.
        self.memory_weight = torch.nn.Parameter(
            data=torch.normal(mean=0.0, std=0.1, size=(self.n_head * self.n_query,)),
            requires_grad=True,
        )

    def calculate_memory_attention(
        self, query: torch.Tensor, memory: torch.Tensor, memory_norm: torch.Tensor
    ) -> torch.Tensor:
        """Computes Equation 7 from https://arxiv.org/pdf/2404.07143 to get the memory component
        of the attention layer, `A_mem`.

        Args:
            query (torch.Tensor):: attention query matrix of shape (n_b, n_seq_q, d_model) where
                d_model = n_query * n_head * d_head
        Returns:
            (torch.Tensor): Memory contribution to the attention, shape
                (n_b, n_seq_q, n_query * n_head, d_head)
        """
        n_b, n_seq_q = query.shape[:2]

        # (n_b, n_seq_q, n_query, n_head * d_head)
        sigma_q = self.mem_activation(
            query.view(n_b, n_seq_q, self.n_query, self.n_head * self.d_head)
        )

        # (n_b, n_seq_q, n_query,  n_head * d_head) @ (d_head * n_head, d_head * n_head) =
        # (n_b, n_seq_q, n_query * self.n_head, self.d_head)
        # Use torch.clone here since we modify self.memory and self.memory_norm in-place during
        # the forward pass, which breaks the backward pass during training.
        num = sigma_q @ memory
        den = sigma_q @ memory_norm
        return torch.div(num, den + 1e-8).view(
            n_b, n_seq_q, self.n_query * self.n_head, self.d_head
        )

    def update_memory(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        memory: torch.Tensor,
        memory_norm: torch.Tensor,
    ) -> None:
        """Apply Equations 8, 9 from https://arxiv.org/pdf/2404.07143 to update the context memory
        of the attention layer.

        Args:
            key (torch.Tensor): attention key matrix of shape (n_b, n_seq_kv, d_head * n_head).
            value (torch.Tensor): attention value matrix of shape (n_b, n_seq_kv, d_head * n_head).
        """
        sigma_k: torch.Tensor = self.mem_activation(key)
        sigma_k_t = sigma_k.transpose(-2, -1)

        batch_update = sigma_k_t @ (
            value - torch.div(sigma_k @ memory, sigma_k @ memory_norm + 1e-8)
        )

        self.memory += torch.sum(batch_update, dim=0)
        self.memory_norm += torch.sum(sigma_k, dim=(0, 1)).unsqueeze(-1)

    def reset_memory(self) -> None:
        """Reset the built-in memory and memory_norm buffers and detach from the current
        computational graph."""
        self.memory.zero_().detach_()
        self.memory_norm.fill_(1.0).detach_()

    def calculate_recurrent_attention(
        self, attention_out: torch.Tensor, attention_memory: torch.Tensor
    ) -> torch.Tensor:
        """Calculates Equation 10 from https://arxiv.org/pdf/2404.07143 to fuse the attention output
        with the memory context using a scalar mixing parameter.

        Args:
            attention_out (torch.Tensor): Attention output tensor `A_out` of shape
                (n_b, n_seq_q, n_query * n_head, d_head).
            attention_memory (torch.Tensor): Attention memory tensor `A_mem` of shape
                (n_b, n_seq_q, n_query * n_head, d_head).
        Returns:
            (torch.Tensor): local and context attention fused output
                (n_b, n_seq_q, n_query * n_head,  d_head).
        """
        mem_weight = torch.nn.functional.sigmoid(self.memory_weight).view(1, 1, -1, 1)
        return attention_out * (1.0 - mem_weight) + attention_memory * mem_weight

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Runs the forward pass, computing `n_head` attention heads with `n_query`
        queries per head. If enabled uses the memory mechanism to mix the local attention
        output with the memory attention.

        Args:
            x (torch.Tensor): Batched tensor of shape (n_b, n_seq_q, d_model)
            y (torch.Tensor): Optional batched tensor of shape (n_b, n_seq_kv, d_model), if present,
                the layer calculates cross-attention between x and y by using x to compute the query
                and y to compute the keys and values.
            mask (torch.Tensor | None) Optional mask of shape (batch_size, 1, n_seq_q, n_seq_kv)
                for casual attention.

        Returns:
            (torch.Tensor): Batched tensor of shape (n_b, n_seq_q, d_model)
        """

        def ckpt_fwd(
            x: torch.Tensor,
            memory: torch.Tensor,
            memory_norm: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

            n_b, n_seq_q, d_model = x.shape

            # (n_b, n_seq_q, n_query * n_head, d_head)
            attention_out, query, key, value = self.calculate_attention_output(
                x=x, y=y, return_qkv=True, mask=mask
            )

            # (n_b, n_seq_q, n_query * n_head, d_head)
            attention_memory = self.calculate_memory_attention(
                query=query, memory=memory, memory_norm=memory_norm
            )

            # Update the memory matrix and normalization term in-place.
            self.update_memory(key=key, value=value, memory=memory, memory_norm=memory_norm)

            # (n_b, n_seq_q, n_query * n_head, d_head) -> ((n_b, n_seq_q, d_model))
            return self.attn_proj(
                self.calculate_recurrent_attention(
                    attention_out=attention_out, attention_memory=attention_memory
                ).reshape(n_b, n_seq_q, d_model)
            )

        return checkpoint(
            ckpt_fwd, x, self.memory.clone(), self.memory_norm.clone(), y, mask, use_reentrant=False
        )
