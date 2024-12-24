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


import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

K_TRAIN_INDICES = [8]  # 0, 1, 2, 3, 4, 5, 6, 7,
K_VAL_INDICES = [9]

# fmt: off
# Lambda values for Yeo-Johnson transformation of the input data.
K_FEATURE_LAMBDAS = [
  0.72179572,  0.96900947,  0.73248581,  0.72192479,  0.99450708,  0.90451684,
  0.95145941,  0.9090044,   0.94620721,  0.27778645,  0.29210153,  0.13750156,
 -0.31383186, -1.38997389, -0.64444204, -0.65741952, -0.80650246, -0.69839666,
  1.02626056,  1.03148867,  0.95597589, -1.6314873,   0.58662938,  0.68307628,
  0.58427999,  0.525378,    0.64749457,  0.91473572,  0.98712525, -0.85465882,
 -0.62961623, -1.61083055,  0.9404928,   1.00092689,  0.90820011,  0.91518583,
  0.99765828,  0.87828082,  0.94987512,  0.92195913,  1.01675781,  0.97505145,
  1.08597373,  0.94462007,  1.00192138,  0.99195147,  1.05831349,  0.92479788,
  1.01287662,  1.07869257,  0.99161477,  0.85450135,  0.92237017,  1.03810044,
  1.0891354 ,  1.09730418,  1.02978006,  0.98764699,  1.05773025,  0.99458541,
  1.04696161,  1.02509799, -1.91452167, -2.11048964, -2.02990744,  0.99449889,
  1.0122254 , -0.43431611, -1.91309719, -0.84537157, -0.25192413, -1.61135546,
 -0.56944075,  0.3366939,   0.33393499,  0.2955401,   0.26717948,  0.29376686,
  0.29398477
]
# fmt: on


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

    def __init__(self, num_epochs: int = 100, train_seq_len: int = 32) -> None:

        self.num_epochs = num_epochs
        self.train_seq_len = train_seq_len

        # Tensor for weighting the elements of the responder loss.
        # Normalized to sum to 1.0.
        self.loss_responder_weighting = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ).cuda()
        self.loss_responder_weighting = torch.div(
            self.loss_responder_weighting, self.loss_responder_weighting.sum()
        )
        print(f"Loss weighting: {self.loss_responder_weighting}")

        self.train_dataloader: DataLoader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_TRAIN_INDICES],
            window_size=32,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.train_dataloader, mode="train")

        # TODO(Bradley) Figure out how to properly do eval so that it doesn't take forever
        # but is also representative of the actual eval conditions. I think we should
        # be able to form sequences of the training length by grabbing the training data
        # and then concatenating it with the first eval time step and marching forward
        # from there.
        self.val_dataloader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_VAL_INDICES],
            window_size=32,
            batch_size=1,
            shuffle=False,
        )

        self.log_dataloader_info(self.val_dataloader, mode="val")

        self.model = self.create_model()
        print(f"Created model with {self.get_num_params()} parameters")

        dummy_inputs: Dict[str, torch.Tensor] = self.make_dummy_input(seq_len=100)
        summary(
            self.model,
            input_data=list(dummy_inputs.values()),
        )

        self.optimizer = self.create_optimizer(self.model)
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=self.optimizer,
            base_lr=self.optimizer.param_groups[0]["lr"],
            max_lr=self.optimizer.param_groups[0]["lr"] * 5.0,
            step_size_up=int(len(self.train_dataloader) * 1.6),
            step_size_down=int(len(self.train_dataloader) * 1.6),
        )

        self.loss = torch.nn.SmoothL1Loss(reduction="none")
        self.yeo_johnson_lambdas: torch.Tensor = torch.tensor(K_FEATURE_LAMBDAS)

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

        # Reset the memory before running an epoch.
        self.model.reset_memory()
        self.optimizer.zero_grad()

        seq_loss: torch.Tensor = torch.zeros(9).cuda()
        epoch_loss: torch.Tensor = torch.zeros((9), requires_grad=False)
        per_responder_mae: torch.Tensor = torch.zeros((9), requires_grad=False)

        total_len: int = 0
        num_rows_per_sequence: int = 0

        for i, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # TODO for training, sample the dataloader to start at different dates so that the model
            # sees slightly different sequences on each epoch

            # Move the sample to the GPU.
            dict_to_cuda(sample=sample)

            with torch.no_grad():
                transformed_features = self.yeo_johnson_transform(sample["features"])

            predictions = self.model.forward(
                date_ids=sample["date_id"],
                time_ids=sample["time_id"],
                symbol_ids=sample["symbol_id"],
                features=transformed_features,
                responders=sample["lags"],
            )

            # Get the targets only for the predictions -- discard the lagged responders from the
            # time step before the first feature input and append the responders at the last
            # feature input.
            targets = torch.cat((sample["lags"], sample["responders_t"]), dim=-2)[
                ..., -predictions.shape[-2] :, :
            ]

            # Compute the loss as the mean of mean of the sequence.
            total_loss: torch.Tensor = self.loss(predictions, targets)

            # Add the total responder vector to the sequence loss.
            seq_loss += total_loss.sum((0, 1))

            # Add to the total length (in terms of DF rows processed) for this
            # sequence.
            total_len += predictions.shape[1] * predictions.shape[0]
            num_rows_per_sequence += predictions.shape[1] * predictions.shape[0]

            # Sum to (n_responder_len,) shape and add to running total
            per_responder_mae += (targets - predictions).abs().sum(dim=(0, 1)).cpu().detach()

            if self.model.training and (i + 1) % self.train_seq_len == 0:
                print("backwards...")
                # Divide the total responder vector by the total number of predicted elements
                # to get the mean responder vector for this sequence.
                seq_loss = torch.div(seq_loss, num_rows_per_sequence)
                epoch_loss += seq_loss.detach().cpu()
                total_seq_loss = (seq_loss * self.loss_responder_weighting).sum()
                print(
                    f"Train Loss: {(seq_loss)}, {total_seq_loss.item()}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']}"
                )

                total_seq_loss.backward()

                # Clip gradients per group.
                for _, param in self.model.named_parameters():
                    torch.nn.utils.clip_grad_norm_(parameters=param, max_norm=1.0)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Reset the model memory matrices and loss for the next sequence.
                self.model.reset_memory()
                seq_loss.zero_().detach_()
                num_rows_per_sequence = 0

        if not self.model.training:
            epoch_loss = torch.div(seq_loss, total_len)
            print(f"Mean responder error: {per_responder_mae / total_len}")
            print(f"Mean Loss: {(epoch_loss)}, {epoch_loss.sum().item()}")

        return per_responder_mae.detach().cpu() / total_len, epoch_loss.detach().cpu()

    def run(self) -> None:
        """Running training and eval."""
        save_path = Path(f"ckpt/{time.time()}")
        save_path.mkdir(parents=True, exist_ok=False)

        torch.save(
            {
                "train_seq_len": self.train_seq_len,
                "train_time_window": self.train_dataloader.dataset.time_context_length,
                "val_time_window": self.train_dataloader.dataset.time_context_length,
                "train_dataloader_len": len(self.train_dataloader),
                "val_dataloader_len": len(self.val_dataloader),
                "lr": self.optimizer.param_groups[0]["lr"],
            },
            f=(save_path / "meta.pt").as_posix(),
        )

        for i in range(self.num_epochs):

            print(f"Training epoch {i}...")
            self.model.train()
            train_mean_responder_error, train_mean_loss_vector = self.run_epoch(
                dataloader=self.train_dataloader
            )

            with torch.no_grad():
                print(f"Validating epoch {i}")
                self.model.reset_memory()
                self.model.eval()

                # Validate on the held-out data.
                val_mean_responder_error, val_mean_loss_vector = self.run_epoch(
                    dataloader=self.val_dataloader
                )

                torch.save(
                    {
                        "epoch": i,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "training": {
                            "mean_responder_error": train_mean_responder_error.detach().cpu(),
                            "mean_loss_vector": train_mean_loss_vector.detach().cpu(),
                        },
                        "validation": {
                            "mean_responder_error": val_mean_responder_error.detach().cpu(),
                            "mean_loss_vector": val_mean_loss_vector.detach().cpu(),
                        },
                        "lr": self.optimizer.param_groups[0]["lr"],
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
            n_blocks=8,
            n_feature_len=79,
            n_responder_len=9,
            n_query=4,
            n_head=4,
            d_model=1024,
        ).cuda()

    @staticmethod
    def create_optimizer(model: torch.nn.Module, lr: float = 0.00025) -> torch.optim.Optimizer:
        """Create the optimizer"""
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
        )

    def get_num_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.shape.numel() for p in self.model.parameters() if p.requires_grad)

    def yeo_johnson_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the Yeo-Johnson transform to the input features. Use the pre-computed lambdas
        that are stored as class members.

        Source: __Yeo, I. K., and Johnson, R. A. (2000). A new family of power transformations to
        improve normality or symmetry

        Args:
            features (torch.Tensor): Input features (..., 79)

        Returns:
            transformed_features (torch.Tensor): Transformed features using the precomputed lambda
                values. Shape (..., 79).
        """

        # TODO unit test this torch implementation against the scipy implementation.
        # Currently I've only tested this visually by comparing the histogram produced by using
        # scipy to transform features against using this to transform the features.

        if self.yeo_johnson_lambdas.device != features.device:
            self.yeo_johnson_lambdas = self.yeo_johnson_lambdas.to(features.device)

        ge_zero_mask: torch.Tensor = features >= 0.0
        lt_zero_mask: torch.Tensor = torch.logical_not(ge_zero_mask)

        output = torch.empty_like(features)

        output[ge_zero_mask] = (
            ((features + 1.0) ** self.yeo_johnson_lambdas - 1.0) / self.yeo_johnson_lambdas
        )[ge_zero_mask]

        lt_zero_exp: torch.Tensor = 2.0 - self.yeo_johnson_lambdas
        output[lt_zero_mask] = (
            -((1.0 - features) ** lt_zero_exp - 1.0) / (2.0 - self.yeo_johnson_lambdas)
        )[lt_zero_mask]

        return output

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

        inputs = self.make_dummy_input(seq_len=100)
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
