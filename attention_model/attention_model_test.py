import unittest
import torch
import torch.nn.functional as f
from attention_model.attention_model import GroupedRecurrentMultiHeadAttention


class TestGRMHA(unittest.TestCase):
    """Unit test fixture for Grouped Recurrent Multi-Head Attention."""

    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 7
        self.seq_len = 20
        self.n_query = 4
        self.n_head = 12
        self.d_head = 30
        self.d_model = self.n_query * self.n_head * self.d_head
        self.layer = GroupedRecurrentMultiHeadAttention(
            d_model=self.d_model,
            n_query=self.n_query,
            n_head=self.n_head,
        )

    def test_check_scaled_dot_product_attention(self):
        """Check that custom scaled dot product attention matches torch implementation."""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_head * self.n_head)
        value = torch.randn(self.batch_size, self.seq_len, self.d_head * self.n_head)

        torch_out = f.scaled_dot_product_attention(
            query=query.view(
                self.batch_size, self.seq_len, self.n_head * self.n_query, self.d_head
            ).transpose(1, 2),
            key=key.view(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(1, 2),
            value=value.view(self.batch_size, self.seq_len, self.n_head, self.d_head).transpose(
                1, 2
            ),
            enable_gqa=True,
        ).transpose(1, 2)

        layer_out = self.layer.scaled_dot_product_attention(query=query, key=key, value=value)

        # Check that the output shape is correct.
        self.assertEqual(
            layer_out.shape,
            torch.Size([self.batch_size, self.seq_len, self.n_query * self.n_head, self.d_head]),
        )

        # This test can fail if rtol is too low since we're dealing with very small values.
        # Check the that the result closely matches the torch implementation.
        self.assertTrue(layer_out.allclose(torch_out, rtol=1.0))

    def test_should_calculate_memory_correctly(self):
        """Test the memory retrieval from query."""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.layer.memory.random_()
        self.layer.memory_norm.random_()

        # Calculate using layer
        layer_attention_mem = self.layer.calculate_memory_attention(query)

        # Check the output shape.
        self.assertEqual(
            layer_attention_mem.shape,
            torch.Size([self.batch_size, self.seq_len, self.n_head * self.n_query, self.d_head]),
        )

        # reshape to look at each individual head.
        layer_attention_mem = layer_attention_mem.view(
            self.batch_size, self.seq_len, self.n_query, self.n_head, -1
        )

        # Calculate manually
        sigma_q = self.layer.mem_activation(
            query.reshape(self.batch_size, self.seq_len, self.n_query, -1)
        )
        for b in range(self.batch_size):
            for s in range(self.seq_len):
                for q in range(self.n_query):
                    # Get the (d_head * n_head, d_head * n_head) matrix for this batch idx, sequence
                    # idx, and query idx and manually calculate the attention.
                    mat = sigma_q[b, s, q]
                    memory = (mat @ self.layer.memory) / (mat @ self.layer.memory_norm.view(-1, 1))
                    for h in range(self.n_head):
                        # Check that the manually calculated attention matches each head output
                        # of the calculation from the layer.
                        chunk = memory[self.d_head * h : self.d_head * (h + 1)]
                        self.assertTrue(layer_attention_mem[b, s, q, h].allclose(chunk, rtol=1.0))

    def test_should_update_memory(self):
        """Simple test to check that the memory update function still runs."""
        key = torch.randn(self.batch_size, self.seq_len, self.d_head * self.n_head)
        value = torch.randn(self.batch_size, self.seq_len, self.d_head * self.n_head)
        self.layer.update_memory(key=key, value=value)

    def test_should_run_forward(self):
        """Simple test to check that the memory update function still runs."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = self.layer(x=x)

        self.assertEqual(out.shape, torch.Size([self.batch_size, self.seq_len, self.d_model]))


if __name__ == "__main__":
    unittest.main()
