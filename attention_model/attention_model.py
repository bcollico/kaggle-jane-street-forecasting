"""Transformer model with Grouped Multi-Head Attention and compressed KV memory.
TODO: There are MHA and Attention implementations in pytorch that I should benchmark against.

Memory mechanism based on Infini-attention: https://arxiv.org/pdf/2404.07143
Grouped-Query Attention based on: https://arxiv.org/pdf/2305.13245
"""

import torch


class GroupedRecurrentMultiHeadAttention(torch.nn.Module):
    """Class for a multi-headed grouped attention layer."""

    def __init__(self, d_model: int, n_query: int, n_head) -> None:
        """
        Initializes the GroupedRecurrentMultiHeadAttention with a single Key/Value projetion and
        multiple query projection matrices.

        Args:
            d_model (int): The size of the input dimension for each vector. For multi-head
                attention, the feature lenght for each query is (`d_model / n_head / n_query`).
            n_out (int): The size of the output dimension for each projection.
            n_query (int): The number of query projections to apply on head head. Must be a factor
                of `d_model`.
            n_head(int): The number of heads in this multi-head attention layer. Must be a factor
                of `d_model`.
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
        self.memory_norm = torch.ones(self.d_head * n_head, 1)

        # Scalar parameter for mixing the attention scores from the current forward pass with the
        # memory context.
        self.memory_weight = torch.nn.Parameter(data=torch.Tensor([1.0]), requires_grad=True)

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
        assert d_model == self.d_model

        # (n_b, n_seq, d_model) @ (d_model, d_head * n_head) = (n_b, n_seq, d_head * n_head)
        key = (self.key_proj(x)).view(n_b, n_seq, self.n_head, self.d_head)

        # (n_b, n_seq, d_model) @ (d_model, d_head * n_head) = (n_b, n_seq, d_head * n_head)
        value = (self.value_proj(x)).view(n_b, n_seq, self.n_head, self.d_head)

        # (n_b, n_seq, d_model) @ (d_model, d_model) = (n_b, n_seq, d_model)
        query = (self.query_proj(x)).view(n_b, n_seq, self.n_query * self.n_head, self.d_head)

        key_t = key.transpose(1, 2).transpose(2, 3).view(n_b, 1, self.n_head, self.d_head, n_seq)
        query_t = query.transpose(1, 2).view(n_b, self.n_query, self.n_head, n_seq, self.d_head)
        value_t = value.transpose(1, 2).view(n_b, 1, self.n_head, n_seq, self.d_head)

        # (n_b, n_query, n_head, n_seq, n_seq)
        attention_scores = query_t @ key_t * self.attention_scale
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # (n_b, n_query, n_head, n_seq, d_head) -> (n_b, n_seq, n_query, n_head, d_head)
        attention_out = (attention_probs @ value_t).transpose(2, 3).transpose(1, 2)

        # (n_b, n_seq, self.n_query * self.n_head, self.d_head) @ (self.d_head * self.n_head, self.d_head * self.n_head)
        # (n_b, n_seq, n_query, self.n_head * self.d_head)
        sigma_q = torch.nn.functional.elu(
            query.view(n_b, n_seq, self.n_query, self.n_head * self.d_head)
        )
        attention_memory = (sigma_q @ self.memory) / (sigma_q @ self.memory_norm)

        # (n_b, n_seq, n_query, n_head, d_head)
        attention_memory = attention_memory.view(n_b, n_seq, self.n_query, self.n_head, self.d_head)

        mem_weight = torch.nn.functional.sigmoid(self.memory_weight)
        attention_out = attention_out * (1.0 - mem_weight) + attention_memory * mem_weight

        return attention_out.reshape(n_b, n_seq, self.d_model)
