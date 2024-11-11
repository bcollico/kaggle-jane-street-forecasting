"""Torch dataset definition."""

from typing import List, Optional

from dataclasses import dataclass

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def _get_cumulative_counts(raw_counts: List[int]) -> np.ndarray:
    """Computes a list of the same length as the input where each element
    is the sum of the elements BEFORE the current element.

    List is returned as a numpy array to support np.searchsorted.
    """
    return np.array([sum(raw_counts[:i]) for i in range(len(raw_counts))])


@dataclass
class RowGroupOffset:
    """Dataclass for managing the index and row offsets to a row group
    within a list of Parquet files."""

    file_idx: int
    row_group_idx: int
    offset: int


@dataclass
class RowGroupCache:
    """POD for holding the currently cached Parquet row group."""

    file_idx: int
    row_group_idx: int
    df: pl.DataFrame


class ParquetDataset(Dataset):
    """Dataset class for loading data from multiple Parquet files.
    Implements batching by row group such that it loads and caches the current
    row group on each sample.

    In combination with a Sampler that fully samples each row group before
    moving on to the next, this limits file I/O and is much faster than naive
    single-row sampling.
    """

    def __init__(self, file_paths: List[str], logging=True):

        self.file_paths = file_paths
        self.file_row_counts: List[int] = []
        self.file_row_group_counts: List[List[int]] = []

        # For each Parquet file, store the number of total rows in the file
        # and the number of rows per row group.
        for f in file_paths:
            pq_file = pq.ParquetFile(f)
            self.file_row_group_counts.append(
                [pq_file.metadata.row_group(i).num_rows for i in range(pq_file.num_row_groups)]
            )
            self.file_row_counts.append(sum(self.file_row_group_counts[-1]))

        self.total_num_rows: int = sum(self.file_row_counts)

        self._setup_cumulative_counts()

        # Member variable for caching a row group.
        self.pq_cache: Optional[RowGroupCache] = None

        if logging:
            # Print the files that were loaded and the total number of samples
            print("Loaded files with rows:")
            for i, file in enumerate(file_paths):
                print(f"\t{self.file_row_counts[i]} : {file}")

            print(f"{len(self)} total samples.")

    def __len__(self) -> int:
        return self.total_num_rows

    def __getitem__(self, idx: int) -> torch.Tensor:
        total_idx = self._calculate_index_from_cumulative_counts(idx, self.cum_total_counts_np)
        group: RowGroupOffset = self.cum_total_counts[total_idx]
        return torch.tensor(
            self._get_single_row_with_row_group_batching(
                file_idx=group.file_idx,
                row_group_idx=group.row_group_idx,
                row_idx=idx - group.offset,
            )
        )

    def _setup_cumulative_counts(self) -> None:
        """Setup the cumulative count arrays using the raw file and row group counts."""
        # Get the cumulative number of rows before the start of each file and each row group
        # within a file.
        self.cum_row_counts: np.ndarray = _get_cumulative_counts(self.file_row_counts)
        self.cum_row_group_counts: List[np.ndarray] = [
            _get_cumulative_counts(c) for c in self.file_row_group_counts
        ]

        # Also keep track of the total offset at each row group paired with
        self.cum_total_counts: List[RowGroupOffset] = []
        for f, cum_row_count in enumerate(self.cum_row_counts):
            for r, cum_row_group_count in enumerate(self.cum_row_group_counts[f]):
                total_offset: int = cum_row_count + cum_row_group_count
                self.cum_total_counts.append(
                    RowGroupOffset(file_idx=f, row_group_idx=r, offset=total_offset)
                )

        # Also store just the offsets in a np array for faster lookup later.
        self.cum_total_counts_np = np.array([v.offset for v in self.cum_total_counts])

    def _get_single_row_with_row_group_batching(
        self, row_idx: int, file_idx: int, row_group_idx: int
    ) -> pl.DataFrame:
        """Get a single row from the parquet file, utilizing row_group cachine
        to minimize file I/O. If the requested row is from the currently
        cached file and row group, get the row directly from the cached
        DataFrame, otherwise, load a new row group into cache and then read.
        """
        if not self.pq_cache or (
            file_idx != self.pq_cache.file_idx or row_group_idx != self.pq_cache.row_group_idx
        ):
            self._load_pq_file(file_idx=file_idx, row_group_idx=row_group_idx)

        return self.pq_cache.df.row(row_idx)

    def _calculate_index_from_cumulative_counts(self, idx: int, counts: np.ndarray) -> int:
        """Find the index of the nearest value in `counts` that is <= idx. Assumes that
        `counts` is monotonically increasing."""
        return np.searchsorted(a=counts, v=idx, side="right") - 1

    def _load_pq_file(self, file_idx: int, row_group_idx: Optional[int] = None) -> None:
        """Load the Parquet file and specified row group as a Polars DataFrame."""

        if (
            self.pq_cache
            and file_idx == self.pq_cache.file_idx
            and row_group_idx == self.pq_cache.row_group_idx
        ):
            # Early return if requesting the cached row group/file.
            return

        f: str = self.file_paths[file_idx]

        # Overwrite the previous Parquet cache
        self.pq_cache = RowGroupCache(
            file_idx=file_idx,
            row_group_idx=row_group_idx,
            df=None,
        )

        # Load the file or row group.
        if row_group_idx is not None:
            self.pq_cache.df = pl.from_arrow(pq.ParquetFile(f).read_row_group(row_group_idx))
        else:
            self.pq_cache.df = pl.read_parquet(f)

        # Simple data cleaning.
        # TODO: Replace with a better strategy for handling missing data.
        self.pq_cache.df = self.pq_cache.df.fill_nan(0.0).fill_null(0.0)
