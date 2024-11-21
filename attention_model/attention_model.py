"""Transformer model with Grouped Multi-Head Attention and compressed KV memory.
TODO: There are MHA and Attention implementations in pytorch that I should benchmark against.

Memory mechanism based on Infini-attention: https://arxiv.org/pdf/2404.07143
Grouped-Query Attention based on: https://arxiv.org/pdf/2305.13245
"""

import torch


class GroupedRecurrentMultiHeadAttention(torch.nn.Module):
    """Class for a multi-headed grouped attention layer."""

    def __init__(
        self,
        d_model: int,
        n_query: int,
        n_head,
        mem_activation: torch.nn.Module = torch.nn.ELU(),
    ) -> None:
        """
        Initializes the GroupedRecurrentMultiHeadAttention with a single Key/Value projetion and
        multiple query projection matrices.

        Args:
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature length for each query is (`d_model / n_head / n_query`).
            n_out (int): The size of the output dimension for each projection.
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head(int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
            mem_activation (torch.nn.Module): Torch module to apply as the `sigma` function when
                calculating the memory component of the attention.
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

        self.mem_activation: torch.nn.Module = mem_activation

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

        # Scaled attention uses the feature length for each query.
        self.attention_scale = 1.0 / torch.sqrt(torch.tensor(self.d_head))

        # Memory parameters. Keep a feature-length memory matrix and normalization vector for each
        # head.
        self.memory = torch.zeros((self.d_head * n_head, self.d_head * n_head))
        self.memory_norm = torch.ones((self.d_head * n_head))

        # Scalar parameter for mixing the attention scores from the current forward pass with the
        # memory context.
        self.memory_weight = torch.nn.Parameter(data=torch.Tensor([1.0]), requires_grad=True)

    def scaled_dot_product_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Apply Equation 6 from https://arxiv.org/pdf/2404.07143 to calculate the `A_dot`
        attention output matrix.

        Args:
            query (torch.Tensor):: attention query matrix of shape (n_b, n_seq, d_model) where
                d_model = n_query * n_head * d_head
            key (torch.Tensor): attention key matrix of shape (n_b, n_seq, d_head * n_head).
            value (torch.Tensor): attention value matrix of shape (n_b, n_seq, d_head * n_head).

        Returns:
            (torch.Tensor): Attention output of shape (n_b, n_seq, n_query * n_head, d_head).
        """
        n_b, n_seq = query.shape[:2]

        # TODO: figure out how to remove the `repeat_interleave` here to compute the
        # grouped attention without copying the key and value data.

        # (n_b, n_head * n_query, d_head, n_seq)
        key_t = (
            key.view(n_b, n_seq, self.n_head, self.d_head)
            .repeat_interleave(self.n_query, -2)
            .transpose(1, 2)
            .transpose(2, 3)
        )

        # (n_b, n_query * n_head, n_seq, d_head)
        query_t = query.view(n_b, n_seq, self.n_query * self.n_head, self.d_head).transpose(1, 2)

        # (n_b, n_query * n_head, n_seq, d_head)
        value_t = (
            value.view(n_b, n_seq, self.n_head, self.d_head)
            .repeat_interleave(self.n_query, -2)
            .transpose(1, 2)
        )

        # (n_b, n_query * n_head, n_seq, n_seq)
        attention_scores = query_t @ key_t * self.attention_scale
        attention_probabilities = torch.nn.functional.softmax(attention_scores, dim=-1)

        # (n_b, n_query * n_head, n_seq, d_head) -> (n_b, n_seq, n_query * n_head, d_head)
        return (attention_probabilities @ value_t).transpose(1, 2)

    def calculate_memory_attention(self, query: torch.Tensor) -> torch.Tensor:
        """Computes Equation 7 from https://arxiv.org/pdf/2404.07143 to get the memory component
        of the attention layer, `A_mem`.

        Args:
            query (torch.Tensor):: attention query matrix of shape (n_b, n_seq, d_model) where
                d_model = n_query * n_head * d_head
        Returns:
            (torch.Tensor): Memory contribution to the attention, shape
                (n_b, n_seq, n_query * n_head, d_head)
        """
        n_b, n_seq = query.shape[:2]

        # (n_b, n_seq, n_query, n_head * d_head)
        sigma_q = self.mem_activation(
            query.view(n_b, n_seq, self.n_query, self.n_head * self.d_head)
        )

        # (n_b, n_seq, n_query,  n_head * d_head) @ (d_head * n_head, d_head * n_head) =
        # (n_b, n_seq, n_query * self.n_head, self.d_head)
        return ((sigma_q @ self.memory) / (sigma_q @ self.memory_norm.view(-1, 1))).view(
            n_b, n_seq, self.n_query * self.n_head, self.d_head
        )

    def update_memory(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Apply Equations 8, 9 from https://arxiv.org/pdf/2404.07143 to update the context memory
        of the attention layer.

        Args:
            key (torch.Tensor): attention key matrix of shape (n_b, n_seq, d_head * n_head).
            value (torch.Tensor): attention value matrix of shape (n_b, n_seq, d_head * n_head).
        """
        sigma_k = self.mem_activation(key)
        sigma_k_t = sigma_k.transpose(-2, -1)

        batch_update = sigma_k_t @ (
            value - ((sigma_k @ self.memory) / (sigma_k @ self.memory_norm.view(-1, 1)))
        )

        self.memory += torch.sum(batch_update, dim=0)
        self.memory_norm += torch.sum(sigma_k, dim=(0, 1))

    def calculate_recurrent_attention(
        self, attention_out: torch.Tensor, attention_memory: torch.Tensor
    ) -> torch.Tensor:
        """Calculates Equation 10 from https://arxiv.org/pdf/2404.07143 to fuse the attention output
        with the memory context using a scalar mixing parameter.

        Args:
            attention_out (torch.Tensor): Attention output tensor `A_out` of shape
                (n_b, n_seq, n_query * n_head, d_head).
            attention_memory (torch.Tensor): Attention memory tensor `A_mem` of shape
                (n_b, n_seq, n_query * n_head, d_head).
        Returns:
            (torch.Tensor): local and context attention fused output
                (n_b, n_seq, n_query * n_head,  d_head).
        """
        mem_weight = torch.nn.functional.sigmoid(self.memory_weight)
        return attention_out * (1.0 - mem_weight) + attention_memory * mem_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass, computing `n_head` attention heads with `n_query`
        queries per head. If enabled uses the memory mechanism to mix the local attention
        output with the memory attention.

        Args:
            x (torch.Tensor): Batched tensor of shape (n_b, n_seq, d_model)
        Returns:
            (torch.Tensor): Batched tensor of shape (n_b, n_seq, d_model)
        """

        # TODO: positional encoding/embedding.

        n_b, n_seq, d_model = x.shape

        # (n_b, n_seq, d_model) @ (d_model, d_head * n_head) = (n_b, n_seq, d_head * n_head)
        key = self.key_proj(x)

        # (n_b, n_seq, d_model) @ (d_model, d_head * n_head) = (n_b, n_seq, d_head * n_head)
        value = self.value_proj(x)

        # (n_b, n_seq, d_model) @ (d_model, d_model) = (n_b, n_seq, d_model)
        query = self.query_proj(x)

        # (n_b, n_seq, n_query * n_head, d_head)
        attention_out = self.scaled_dot_product_attention(query=query, key=key, value=value)

        # (n_b, n_seq, n_query * n_head, d_head)
        attention_memory = self.calculate_memory_attention(query=query)

        # Update the memory matrix and normalization term in-place.
        self.update_memory(key=key, value=value)

        # (n_b, n_seq, n_query * n_head, d_head) -> ((n_b, n_seq, d_model))
        return self.calculate_recurrent_attention(
            attention_out=attention_out, attention_memory=attention_memory
        ).reshape(n_b, n_seq, d_model)
