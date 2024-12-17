from typing import Any, Dict, List, OrderedDict
from collections import OrderedDict
import tqdm
from pathlib import Path
import time

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

K_TRAIN_INDICES = [8]  # 0, 1, 2, 3, 4, 5, 6, 7,
K_VAL_INDICES = [9]


def make_parquet_path(i: int) -> str:
    """Create a string path to a parquet file."""
    return (
        f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/"
        f"partition_id={i}/part-0.parquet"
    )


def dict_to_cuda(sample: Dict[Any, torch.Tensor]) -> None:
    """In-place move tensors in a dictionary to CUDA."""
    for k, v in sample.items():
        sample[k] = v.cuda()


class ModelRunner:
    """Model running class for training."""

    def __init__(self, num_epochs: int = 1000, train_seq_len: int = 1) -> None:

        self.num_epochs = num_epochs
        self.train_seq_len = train_seq_len

        self.train_dataloader: DataLoader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_TRAIN_INDICES],
            window_size=64,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.train_dataloader, mode="train")

        # At true evaluation time, the window size will be 1, but it's pretty slow to eval with a
        # window size of 1 during training, so set to 64 to capture the general trends (or maybe
        # I should just limit the length of this dataloader.)
        self.val_dataloader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_VAL_INDICES],
            window_size=64,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.val_dataloader, mode="val")

        self.model = self.create_model()
        print(f"Created model with {self.get_num_params()} parameters")

        dummy_inputs: Dict[str, torch.Tensor] = self.make_dummy_input()
        summary(
            self.model,
            input_data=list(dummy_inputs.values()),
        )

        self.optimizer = self.create_optimizer(self.model)

        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def make_dummy_input(seq_len: int = 6000) -> OrderedDict[str, torch.Tensor]:
        """Make dummy input tensors to the model. Output in ordered dict so that they can be used
        with or without keyword arguments."""

        return OrderedDict(
            {
                "date_ids": torch.randint(0, 1500, (1, seq_len)).long(),
                "time_ids": torch.randint(0, 1000, (1, seq_len)).long(),
                "symbol_ids": torch.randint(0, 256, (1, seq_len)).long(),
                "features": torch.randn((1, seq_len, 79)).float(),
                "lags": torch.randn((1, seq_len, 9)).float(),
            }
        )

    def run_epoch(self, dataloader: DataLoader) -> None:
        """Run the model through the dataloader one full time."""

        # Always reset the memory before running an epoch
        self.model.reset_memory()

        seq_loss: torch.Tensor = torch.tensor(0.0).cuda()
        per_responder_mae: torch.Tensor = torch.zeros((9), requires_grad=False)
        total_len: int = 0

        for i, sample in enumerate(dataloader):
            # TODO for training, sample the dataloader to start at different dates so that the model
            # sees slightly different sequences on each epoch

            # Move the sample to the GPU.
            dict_to_cuda(sample=sample)

            predictions = self.model.forward(
                date_ids=sample["date_id"],
                time_ids=sample["time_id"],
                symbol_ids=sample["symbol_id"],
                features=sample["features"],
                responders=sample["lags"],
            )

            # Get the targets only for the predictions -- discard the lagged responders from the
            # time step before the first feature input and append the responders at the last
            # feature input.
            targets = torch.cat((sample["lags"], sample["responders_t"]), dim=-2)[
                ..., -predictions.shape[-2] :, :
            ]

            # Compute the loss, masking the responders that are NaN or Null for computing the Loss
            total_loss: torch.Tensor = self.loss(predictions, targets)
            seq_loss += total_loss.mean()

            # Add to the total length (in terms of DF rows processed) for this
            # sequence.
            total_len += predictions.shape[1]

            if self.model.training and (i + 1) % self.train_seq_len == 0:
                print(seq_loss / self.train_seq_len)
                seq_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Reset the model memory matrices and loss for the next sequence.
                self.model.reset_memory()
                seq_loss = seq_loss.zero_().detach()

                total_len = 0
                break

            elif not self.model.training:
                # Sum to (n_responder_len,) shape and add to running total
                per_responder_mae += (targets - predictions).abs().cpu().detach().sum(dim=(0, 1))

        if not self.model.training:
            print(f"Mean responder error: {per_responder_mae / total_len}")

    def run(self) -> None:
        """Running training and eval."""
        save_path = Path(f"ckpt/{time.time()}")
        save_path.mkdir(parents=True, exist_ok=False)

        for i in range(self.num_epochs):
            print(f"Training epoch {i}...")
            self.model.train()
            self.run_epoch(dataloader=self.train_dataloader)

            with torch.no_grad():
                print(f"Validating epoch {i}")
                self.model.eval()
                # Create the full memory context of the training set
                # self.run_epoch(dataloader=self.train_dataloader)
                # Validate on the held-out data.
                self.run_epoch(dataloader=self.val_dataloader)

            torch.save(
                {
                    "epoch": i,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                f=(save_path / f"{i}.pt").as_posix(),
            )

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
            n_blocks=1,
            n_feature_len=79,
            n_responder_len=9,
            n_query=4,
            n_head=1,
            d_model=256,
        ).cuda()

    @staticmethod
    def create_optimizer(model: torch.nn.Module, lr: float = 0.001) -> torch.optim.Optimizer:
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

    def profile(self) -> None:
        """Run pytorch CPU and GPU profiler on the model and print to console."""
        from torch.profiler import profile, ProfilerActivity

        inputs = self.make_dummy_input(seq_len=1000)
        dict_to_cuda(sample=inputs)

        print("Running profiler...")

        self.model.train()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            out: torch.Tensor = self.model(**inputs)
            out.mean().backward()

        print(prof.key_averages().table(sort_by="cuda_memory_usage"))


def scalene_profile():
    """Simple runner to profile a training step. To be invoked with Scalene profiler. via
    `scalene train/train.py` from command line.
    """
    model = ModelRunner.create_model()
    inputs = ModelRunner.make_dummy_input(seq_len=1000)
    dict_to_cuda(inputs)

    out: torch.Tensor = model(**inputs)
    out.backward()


if __name__ == "__main__":
    runner = ModelRunner()
    runner.run()
