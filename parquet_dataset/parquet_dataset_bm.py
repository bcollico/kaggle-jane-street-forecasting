"""Simple timing benchmark for the parquet dataset."""

from typing import List

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from parquet_dataset.parquet_dataset import ParquetDataset
from utils.time import timing


@timing
def iterate_samples_batched_row_group(pq_dataset: ParquetDataset, step_size: int):
    """Timing benchmark for the ParquetDataset. Requires that the kaggle data is downloaded to
    /kaggle/input/.
    """

    for idx in range(0, len(pq_dataset), step_size):
        sample = pq_dataset[idx]


def make_train_parquet_path(i: int) -> str:
    """Create a string path to a parquet file."""
    return f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet"


if __name__ == "__main__":
    # Setup the file indices to use.
    n_train_files: int = 10
    train_files: List[str] = [make_train_parquet_path(i) for i in range(n_train_files)]
    dataset = ParquetDataset(file_paths=train_files, logging=False)

    print(f"Running with {len(dataset)} samples...")

    iterate_samples_batched_row_group(pq_dataset=dataset, step_size=100000)
    iterate_samples_batched_row_group(pq_dataset=dataset, step_size=1000)
    iterate_samples_batched_row_group(pq_dataset=dataset, step_size=100)
