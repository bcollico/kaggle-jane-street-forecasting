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


class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.grad_norms = {}
        # self.register_hooks()

    # def register_hooks(self):
    #     for name, param in self.model.named_parameters():
    #         param.register_hook(lambda grad, name=name: self.record_grad_norm(name, grad))

    # def record_grad_norm(self, name, grad):
    #     self.grad_norms[name] = grad.norm().item()

    def print_gradient_report(self):
        for name, param in self.model.named_parameters():
            self.grad_norms[name] = param.grad.norm().item()

        print_summary = False
        # Detect potential issues
        norms = list(self.grad_norms.values())
        if any(norm > 1000 for norm in norms):
            print_summary = True
            print("WARNING: Potential Exploding Gradients")
        if any(norm < 1e-6 for norm in norms):
            print_summary = True
            print("WARNING: Potential Vanishing Gradients")

        if print_summary:
            for name, norm in self.grad_norms.items():
                print(f"{name}: {norm}")


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

    def __init__(self, num_epochs: int = 500, train_seq_len: int = 128) -> None:

        self.num_epochs = num_epochs
        self.train_seq_len = train_seq_len

        self.train_dataloader: DataLoader = self.create_dataloader(
            parquet_files=[make_parquet_path(i) for i in K_TRAIN_INDICES],
            window_size=32,
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
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epochs,  # * len(self.train_dataloader) / 10
            eta_min=self.optimizer.param_groups[0]["lr"] / 10.0,
        )

        self.loss = torch.nn.SmoothL1Loss(reduction="none")
        self.monitor = GradientMonitor(model=self.model)
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

        # for i, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        for i, sample in enumerate(dataloader):
            # TODO for training, sample the dataloader to start at different dates so that the model
            # sees slightly different sequences on each epoch

            # Move the sample to the GPU.
            dict_to_cuda(sample=sample)

            with torch.no_grad():
                transformed_features = self.yeo_johnson_transform(sample["features"])
                # transformed_features = sample["features"]

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

            # Compute the loss, masking the responders that are NaN or Null for computing the Loss
            total_loss: torch.Tensor = self.loss(predictions, targets)
            seq_loss += total_loss.mean()

            # Add to the total length (in terms of DF rows processed) for this
            # sequence.
            total_len += predictions.shape[1]

            if self.model.training and (i + 1) % self.train_seq_len == 0:
                print(
                    f"Train Loss: {(seq_loss / self.train_seq_len).item()}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']}"
                )

                seq_loss.backward()

                for _, param in self.model.named_parameters():
                    torch.nn.utils.clip_grad_norm_(parameters=param, max_norm=0.5)

                self.monitor.print_gradient_report()

                self.optimizer.step()
                self.lr_scheduler.step()
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
            print(f"Mean loss: {(seq_loss / total_len).item()}")

    def run(self) -> None:
        """Running training and eval."""
        save_path = Path(f"ckpt/{time.time()}")
        save_path.mkdir(parents=True, exist_ok=False)

        for i in range(self.num_epochs):
            print(f"Training epoch {i}...")
            self.model.train()
            self.run_epoch(dataloader=self.train_dataloader)

            # with torch.no_grad():
            # print(f"Validating epoch {i}")
            # self.model.eval()

            # Create the full memory context of the training set
            # self.run_epoch(dataloader=self.train_dataloader)

            # Validate on the held-out data.
            # self.run_epoch(dataloader=self.val_dataloader)

            # torch.save(
            #     {
            #         "epoch": i,
            #         "state_dict": self.model.state_dict(),
            #         "optimizer": self.optimizer.state_dict(),
            #     },
            #     f=(save_path / f"{i}.pt").as_posix(),
            # )

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
