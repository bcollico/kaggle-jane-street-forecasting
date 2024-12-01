"""Preprocess the parquet files into chunked torch tensors for each symbol ID. Maintain separate
tensors for features (incl. time info) and responders."""

import concurrent.futures
from typing import List, Tuple
from pathlib import Path

from collections import defaultdict

import polars as pl
import pyarrow
import pyarrow.dataset
import pyarrow.parquet
import torch
import tqdm
import numpy as np
import psutil

import concurrent
from concurrent.futures import ThreadPoolExecutor

K_OUT_PATH = Path("./data")
K_CHUNK_SIZE = 1024


def print_ram():
    # Get memory usage of the current process
    process = psutil.Process()
    memory_info = process.memory_info()
    rss = memory_info.rss  # Resident Set Size (memory used by the process)

    print(f"Memory usage: {rss / (1024 * 1024):.2f} MB")


def make_train_parquet_path(i: int) -> str:
    return f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet"


# Setup the file indices to use.
K_MAX_TRAIN_FILES: int = 10
K_TRAIN_FILES: List[str] = [make_train_parquet_path(i) for i in range(K_MAX_TRAIN_FILES)]

K_FEATURE_LEN = 83
K_RESPONDER_LEN = 9


def get_features_and_responders_for_symbol(
    symbol_df: pl.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # symbol_df = df.filter(pl.col("symbol_id") == symbol)
    features = (
        symbol_df.select(
            [col for col in symbol_df.columns if ("responder" not in col and col != "partition_id")]
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_torch()
    )
    responders = (
        symbol_df.select(
            [col for col in symbol_df.columns if ("responder" in col and col != "partition_id")]
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_torch()
    )

    return features, responders


def save_chunked_tensors(
    features: torch.Tensor, responders: torch.Tensor, path: Path, chunk_size: int, symbol_id: int
) -> None:

    print(f"Saving chunked tensors for symbol {symbol_id}")
    n_chunks = int(np.ceil(features.shape[0] / chunk_size) + 0.5)

    symbol_dir: Path = path / str(symbol_id)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    start_idx = 0
    end_idx = chunk_size
    for i in tqdm.tqdm(range(n_chunks)):
        feature_path: Path = (symbol_dir / f"features_{str(i).zfill(4)}").with_suffix(".pt")
        responder_path: Path = (symbol_dir / f"responders_{str(i).zfill(4)}").with_suffix(".pt")

        torch.save(torch.clone(features[start_idx:end_idx]).float(), feature_path)
        torch.save(torch.clone(responders[start_idx:end_idx]).float(), responder_path)

        start_idx += chunk_size
        end_idx += chunk_size


def fcn(symbol):
    df = pl.from_arrow(
        pyarrow.parquet.ParquetDataset(
            "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/",
            filters=[("symbol_id", "=", symbol)],
        ).read()
    )

    features, responders = get_features_and_responders_for_symbol(symbol_df=df)
    save_chunked_tensors(
        features=features,
        responders=responders,
        path=K_OUT_PATH,
        chunk_size=K_CHUNK_SIZE,
        symbol_id=symbol,
    )


def process_parquets():

    dataset = pyarrow.parquet.ParquetDataset(
        "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/"
    )
    # print(dataset.files)

    symbols = pl.from_arrow(dataset.read(["symbol_id"])).get_column("symbol_id").unique().to_list()

    # print(symbols)
    # print_ram()

    # dt_df = pl.from_arrow(dataset.read(["date_id", "time_id"]))

    # print_ram()

    # date ids are in order and dense
    # all time ids are in order and dense within a date.
    # store the number of timesteps at each date
    # times: List[List[int]] = []
    # max_date = max(dt_df.get_column("date_id").unique().to_list())
    # for i in range(max_date+1):
    #     times.append(
    #         max(dt_df.filter(pl.col("date_id") == i).get_column("time_id").unique().to_list())+1
    #     )

    # print_ram()

    # datasets = [make_dataset(i).read(["date_id", "time_id"]) for i in symbols]

    print("Found symbols:", symbols)

    for i in symbols:
        fcn(i)

if __name__ == "__main__":
    process_parquets()
