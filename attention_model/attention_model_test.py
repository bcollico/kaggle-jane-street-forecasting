import unittest
import torch

from attention_model.attention_model import TransformerBlock, TransformerModel
from attention_layers.attention_layers import RotaryPositionalEncoding


class ModelTestFixture(unittest.TestCase):

    def setUp(self):
        self.d_head: int = 64
        self.n_head: int = 8
        self.n_query: int = 2
        self.d_model: int = self.d_head * self.n_head * self.n_query

        self.rope = RotaryPositionalEncoding(d_model=self.d_model)

    def test_should_run_block(self):
        """Simple test to make sure that the transformer block can be initialized and put through
        the forward pass."""
        block = TransformerBlock(
            d_model=self.d_model,
            n_head=self.n_head,
            n_query=self.n_query,
            rope=self.rope,
        )

        batch_size = 10
        seq_len = 64
        x = torch.randn(batch_size, seq_len, self.d_model)

        block.forward(x=x)

    def test_should_run_model(self):
        """Simple test to make sure that the transformer block can be initialized and put through
        the forward pass."""
        n_feature_len: int = 32
        n_responder_len: int = 16
        model = TransformerModel(
            d_model=self.d_model,
            n_head=self.n_head,
            n_query=self.n_query,
            n_blocks=8,
            n_output_bins=100,
            n_feature_len=n_feature_len,
            n_responder_len=n_responder_len,
        )

        batch_size = 2  
        seq_len = 64
        features = torch.randn(batch_size, seq_len, n_feature_len)
        responders = torch.randn(batch_size, seq_len, n_responder_len)
        symbol_ids = torch.randint(0, 255, (batch_size, seq_len, 1))
        time_ids = torch.randint(0, 1000, (batch_size, seq_len, 1))
        date_ids = torch.randint(0, 8, (batch_size, seq_len, 1))

        model.forward(
            features=features,
            responders=responders,
            time_ids=time_ids,
            date_ids=date_ids,
            symbol_ids=symbol_ids,
        )
