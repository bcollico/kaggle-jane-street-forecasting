import unittest
import torch
import torch.nn.functional as f
from attention_model.attention_model import GroupedRecurrentMultiHeadAttention


class TestGRMHA(unittest.TestCase):
    """Unit test fixture for Grouped Recurrent Multi-Head Attention."""

    def setUp(self):
        self.n_query = 2
        self.n_head = 3
        self.d_head = 4
        self.d_model = self.n_query * self.n_head * self.d_head
        self.layer = GroupedRecurrentMultiHeadAttention(
            d_model=self.d_model,
            n_query=self.n_query,
            n_head=self.n_head,
        )

    def test_check_scaled_dot_product_attention(self):
        batch_size = 2
        seq_len = 5
        query = torch.randn(batch_size, seq_len, self.d_model)
        key = torch.randn(batch_size, seq_len, self.d_head * self.n_head)
        value = torch.randn(batch_size, seq_len, self.d_head * self.n_head)

        torch_out = f.scaled_dot_product_attention(
            query=query.reshape(batch_size, seq_len, self.n_head * self.n_query, -1).transpose(
                1, 2
            ),
            key=key.reshape(batch_size, seq_len, self.n_head, -1).transpose(1, 2),
            value=value.reshape(batch_size, seq_len, self.n_head, -1).transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)

        layer_out = self.layer.scaled_dot_product_attention(query=query, key=key, value=value)

        # The order of the output matrices will be out of order compared to the torch
        # implementation since they use `torch.repeat_interleave` whereas my implementation
        # takes more of a `torch.tile` approach.


        for torch_idx in range(self.n_head * self.n_query):
            out = []
            for layer_idx in range(self.n_head * self.n_query):
                out.append(torch.norm(layer_out[:,:,layer_idx,:] - (torch_out[:,:,torch_idx,:])).item())
            print(min(out))


if __name__ == "__main__":
    unittest.main()
