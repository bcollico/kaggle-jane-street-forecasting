from typing import List

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from attention_model.attention_model import TransformerModel
from parquet_dataset.parquet_dataset import DatetimeParquetDataset

K_TRAIN_INDICES = [0]  # , 1, 2, 3, 4, 5, 6, 7, 8]
K_VAL_INDICES = [9]


def make_parquet_path(i: int) -> str:
    """Create a string path to a parquet file."""
    return (
        f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/"
        f"partition_id={i}/part-0.parquet"
    )


class ModelRunner:
    """Model running class for training."""

    def __init__(self, num_epochs: int = 10, train_seq_len: int = 32) -> None:

        self.num_epochs = num_epochs
        self.train_seq_len = train_seq_len

        self.train_dataloader: DataLoader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_TRAIN_INDICES],
            window_size=64,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.train_dataloader, mode="train")

        self.val_dataloader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_VAL_INDICES],
            window_size=64,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.val_dataloader, mode="val")

        self.model = self.create_model()
        print(f"Created model with {self.get_num_params()} parameters")
        summary(
            self.model,
            input_size=[(1, 6000, 1), (1, 6000, 1), (1, 6000, 1), (1, 6000, 79), (1, 6000, 9)],
            dtypes=[torch.int64, torch.int64, torch.int64, torch.float32, torch.float32],
        )

        self.optimizer = self.create_optimizer(self.model)

        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def run_epoch(self, dataloader: DataLoader) -> None:
        """Run the model through the dataloader one full time."""

        seq_loss = torch.tensor(0.0).cuda()

        for i, sample in enumerate(dataloader):
            print(i)

            # Move the sample to the GPU.
            for k, v in sample.items():
                sample[k] = v.cuda()

            predictions = self.model.forward(
                date_ids=sample["date_id"],
                time_ids=sample["time_id"],
                symbol_ids=sample["symbol_id"],
                features=sample["features"],
                responders=sample["responders"],
            )

            total_loss: torch.Tensor = self.loss(predictions, sample["responders"])

            seq_loss += total_loss.mean()

            if self.model.training and (i + 1) % self.train_seq_len == 0:
                print(seq_loss / self.train_seq_len)

                seq_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.model.reset_memory()

                seq_loss = seq_loss.zero_().detach()

    def run(self) -> None:
        """Running training and eval."""
        for _ in range(self.num_epochs):
            self.model.train()
            self.run_epoch(dataloader=self.train_dataloader)

            with torch.no_grad():
                self.model.eval()
                self.run_epoch(dataloader=self.val_dataloader)

    @staticmethod
    def create_dataloader(parquet_files: List[str], window_size: int, **kwargs) -> DataLoader:
        """Create the parquet dataloader. Pass DataLoader options as kwargs."""
        dataset = DatetimeParquetDataset(
            file_paths=parquet_files, time_context_length=window_size, logging=False
        )
        return DataLoader(dataset=dataset, **kwargs)

    @staticmethod
    def create_model() -> torch.nn.Module:
        """Create the model on GPU."""
        return TransformerModel(
            n_blocks=8,
            n_feature_len=79,
            n_responder_len=9,
            n_query=4,
            n_head=4,
            d_model=1024,
        ).cuda()

    @staticmethod
    def create_optimizer(model: torch.nn.Module, lr: float = 0.0001) -> torch.optim.Optimizer:
        """Create the optimizer"""
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
        )

    def get_num_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.shape.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def log_dataloader_info(dataloader: DataLoader, mode: str) -> None:
        """Log the number of files and samples in each dataloader."""
        print(
            f"Created {mode} dataloader with {len(dataloader.dataset.file_paths)} files"
            f" and {len(dataloader)} samples"
        )


if __name__ == "__main__":
    runner = ModelRunner()
    runner.run()
