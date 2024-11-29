"""Preprocess the parquet files into chunked torch tensors for each symbol ID. Maintain separate
tensors for features (incl. time info) and responders."""

from typing import List, Dict, Tuple
import polars as pl

import torch


def make_train_parquet_path(i: int) -> str:
    return f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet"


# Setup the file indices to use.
K_MAX_TRAIN_FILES: int = 10
K_TRAIN_FILES: List[str] = [make_train_parquet_path(i) for i in range(K_MAX_TRAIN_FILES)]


def process_parquets():
    tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for file in K_TRAIN_FILES[:3]:
        df = pl.read_parquet(file)
        symbols = df.get_column("symbol_id").unique().to_list()
        print(symbols)
        print(df.shape)
        for symbol in symbols:
            symbol_df = df.filter(pl.col("symbol_id") == symbol)
            features = (
                symbol_df.select([col for col in df.columns if "responder" not in col])
                .fill_nan(0.0)
                .fill_null(0.0)
                .to_torch()
            )
            responders = (
                symbol_df.select([col for col in df.columns if "responder" in col])
                .fill_nan(0.0)
                .fill_null(0.0)
                .to_torch()
            )

            int_symbol = int(symbol)

            if int_symbol not in tensors:
                tensors[int_symbol] = (features, responders)
            else:
                f, r = tensors[int_symbol]

                tensors[int_symbol] = (torch.vstack((f, features)), torch.vstack((r, responders)))

    for s, (t1, t2) in tensors.items():
        print(s, t1.shape, t2.shape)


if __name__ == "__main__":
    process_parquets()
