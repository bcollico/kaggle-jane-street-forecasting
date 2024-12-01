from typing import List

import torch
from torch.utils.data import DataLoader

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from attention_model.attention_model import TransformerModel
from parquet_dataset.parquet_dataset import DatetimeParquetDataset


def make_train_parquet_path(i: int) -> str:
    """Create a string path to a parquet file."""
    return f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet"


def create_dataloader(parquet_files: List[str], window_size: int, **kwargs) -> DataLoader:
    """Create the parquet dataloader. Pass DataLoader options as kwargs."""
    dataset = DatetimeParquetDataset(file_paths=parquet_files, time_context_length=window_size)
    return DataLoader(dataset=dataset, **kwargs)


def create_model() -> TransformerModel:
    """Create the model."""
    return TransformerModel(
        n_blocks=1,
        n_feature_len=79,
        n_responder_len=9,
        n_query=8,
        n_head=4,
        n_output_bins=101,
        d_model=1024,
    )


def create_optimizer(model: torch.nn.Module, lr: float = 0.0001) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
    )


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.shape.numel() for p in model.parameters() if p.requires_grad)


def train(num_epochs: int = 100) -> None:
    """Train a model."""
    dataloader = create_dataloader(
        parquet_files=[make_train_parquet_path(i) for i in range(1)],
        window_size=128,
        batch_size=1,
        # num_workers=6,
        shuffle=False,
        # prefetch_factor=6,
    )

    print(f"Created dataloader with {len(dataloader)} samples")

    model = create_model().cuda()
    optim = create_optimizer(model=model)

    print(f"Created model with {get_num_params(model)} parameters.")

    nll_loss = torch.nn.NLLLoss()
    # l2_loss = torch.nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, sample in enumerate(dataloader):
            if i == 0:
                # TODO remove this hack. This avoids the problem where
                # we sample only the first timestep and end up with NaN values
                # because we can't attend to anything in the first cross attention block.
                continue
            with torch.autograd.detect_anomaly():
                optim.zero_grad()

                responders = sample["responders"].cuda()

                pred_probs, _ = model.forward(
                    date_ids=sample["date_id"].cuda(),
                    time_ids=sample["time_id"].cuda(),
                    symbol_ids=sample["symbol_id"].cuda(),
                    features=sample["features"].cuda(),
                    responders=responders,
                )

                targets = (100 * ((responders + 5.0) / 10.0)).long()

                loss = nll_loss.forward(
                    input=pred_probs.permute(0, 3, 1, 2).contiguous(), target=targets
                )
                loss.backward()
                optim.step()

                print(loss.item())

                if i > 4:
                    break


if __name__ == "__main__":
    train()
