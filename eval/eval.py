from typing import Optional
from pathlib import Path

import torch
import polars as pl

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from train.train import ModelRunner


def make_parquet_path(i: int) -> str:
    """Create a string path to a parquet file."""
    return (
        f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/"
        f"partition_id={i}/part-0.parquet"
    )


class Evaluator:
    def __init__(self, ckpt: Path):
        self.window_size: int = 16
        self.runner = ModelRunner(window_size=self.window_size)

        self.checkpoint = torch.load(ckpt)
        self.runner.model.load_state_dict(self.checkpoint["state_dict"])

        self.lags: Optional[pl.DataFrame] = None

    def create_memory_context(self):
        # Setup the memory context using the full training set and report the mean errors.
        with torch.no_grad():
            self.runner.model.eval()
            self.runner.model.reset_memory()
            mae, _ = self.runner.run_epoch(
                dataloader=self.runner.val_dataloader,
                train_seq_len=self.runner.train_seq_len,
            )

        print("MAE: ", mae)

    def predict(self, test: pl.DataFrame, lags: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        if lags is not None:
            self.lags = lags

        # Lagged Responders. Date IDs are advanced by 1 to match them with their associated feature.
        lagged_df: torch.Tensor = torch.from_numpy(test.fill_nan(0.0).fill_null(0.0).to_numpy())
        date_ids: torch.Tensor = lagged_df[[0]].int() - 1
        time_ids: torch.Tensor = lagged_df[[1]].int()
        symbol_ids: torch.Tensor = lagged_df[[2]].int()
        responders: torch.Tensor = lagged_df[3:].float()

        # Current date and time
        current_df: torch.Tensor = torch.from_numpy(test.fill_nan(0.0).fill_null(0.0).to_numpy())
        date_ids = torch.vstack((date_ids, current_df[[1]]))
        time_ids = torch.vstack((time_ids, current_df[[2]]))
        symbol_ids = torch.vstack((symbol_ids, current_df[[3]]))
        features: torch.Tensor = current_df[6:]

        predictions: torch.Tensor = self.runner.model.forward(
            date_ids=date_ids.cuda().unsqueeze(0),
            symbol_ids=symbol_ids.cuda().unsqueeze(0),
            time_ids=time_ids.cuda().unsqueeze(0),
            features=features.cuda().unsqueeze(0),
            responders=responders.cuda().unsqueeze(0),
        )

        return test.select("row_id").with_columns(pl.Series("responder_6", predictions[0, :, 6]))
