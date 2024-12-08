from typing import Any, Dict, List, OrderedDict
from collections import OrderedDict
import tqdm

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


def dict_to_cuda(sample: Dict[Any, torch.Tensor]) -> None:
    """In-place move tensors in a dictionary to CUDA."""
    for k, v in sample.items():
        sample[k] = v.cuda()


class ModelRunner:
    """Model running class for training."""

    def __init__(self, num_epochs: int = 10, train_seq_len: int = 128) -> None:

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
                "responders": torch.randn((1, seq_len, 9)).float(),
            }
        )

    def run_epoch(self, dataloader: DataLoader) -> None:
        """Run the model through the dataloader one full time."""

        seq_loss: torch.Tensor = torch.tensor(0.0).cuda()
        per_responder_loss: torch.Tensor = torch.zeros((9), requires_grad=False)
        total_len: int = 0

        for i, sample in tqdm.tqdm(enumerate(dataloader)):
            # TODO for training, sample the dataloader to start at different dates so that the model
            # sees slightly different sequences on each epoch

            # Move the sample to the GPU.
            dict_to_cuda(sample=sample)

            predictions = self.model.forward(
                date_ids=sample["date_id"],
                time_ids=sample["time_id"],
                symbol_ids=sample["symbol_id"],
                features=sample["features"],
                responders=sample["responders"],
            )

            total_loss: torch.Tensor = self.loss(predictions, sample["responders"])
            total_len += predictions.shape[1]
            seq_loss += total_loss.mean()

            # Sum to (n_responder_len,) shape and add to running total
            per_responder_loss += total_loss.cpu().detach().sum(dim=(0, 1))

            if self.model.training and (i + 1) % self.train_seq_len == 0:

                seq_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Reset the model memory matrices for the next sequence.
                self.model.reset_memory()

                seq_loss = seq_loss.zero_().detach()

                total_len = 0
                per_responder_loss.zero_()

        if not self.model.training:
            print(f"Mean responder error: {per_responder_loss / total_len}")

    def run(self) -> None:
        """Running training and eval."""
        with torch.no_grad():
                print("Running baseline validation epoch...")
                self.model.eval()
                self.run_epoch(dataloader=self.val_dataloader)

        for i in range(self.num_epochs):
            print(f"Training epoch {i}...")
            self.model.train()
            self.run_epoch(dataloader=self.train_dataloader)

            with torch.no_grad():
                print(f"Validating epoch {i}")
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
            n_blocks=1,
            n_feature_len=79,
            n_responder_len=9,
            n_query=4,
            n_head=4,
            d_model=512,
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
