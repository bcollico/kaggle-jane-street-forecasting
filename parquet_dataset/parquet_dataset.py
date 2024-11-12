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

            print(f"{self.total_num_rows} total rows.")

    def __len__(self) -> int:
        return self.total_num_rows

    def __getitem__(self, idx: int) -> torch.Tensor:
        total_idx = self.calculate_index_from_cumulative_counts(idx, self.cum_total_counts_np)
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

    def calculate_index_from_cumulative_counts(self, idx: int, counts: np.ndarray) -> int:
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


@dataclass
class DateTimeSegment:
    # List of file/rowgroup segment ids for this unique datetime
    date: int
    time: int
    start_offset: int
    end_offset: int
    start_idx: int
    end_idx: int


class DatetimeParquetDataset(ParquetDataset):
    """Parquet dataset where each sample is a unique date+time with multiple rows (corresponding to
    multiple symbols). The samples will also include features+lags for the time_context_window
    previous date+time samples."""

    def __init__(self, file_paths: List[str], time_context_length: int = 1, logging=True):
        super().__init__(file_paths=file_paths, logging=logging)

        self.time_context_length = time_context_length

        # Go through all of the parquet files again and store the start row for each date+time.
        self.date_time_segments: List[DateTimeSegment] = []

        # Iterate forward through the Parquet files and partially fill the DateTimeSegments.
        self._populate_partial_dt_segments()

        # Iterate through all the partial DateTimeSegments and finish populating
        # them with offsets and segment indices.
        self._finalize_dt_segments()

    def _populate_partial_dt_segments(self) -> None:
        prev_date, prev_time = None, None
        for file_idx, file in enumerate(self.file_paths):

            # load the ith file and get just the date and time columns
            pq_df: pl.DataFrame = pl.read_parquet(file).select(["date_id", "time_id"])

            # overall row where this file starts
            start_row = self.cum_row_counts[file_idx]

            for row_idx in range(len(pq_df)):
                date, time = pq_df.row(row_idx)

                if prev_date is None or prev_time is None or date != prev_date or time != prev_time:
                    # Create a segment at the start of the new date+time
                    segment = DateTimeSegment(
                        date=date,
                        time=time,
                        start_offset=start_row + row_idx,
                        end_offset=None,
                        start_idx=None,
                        end_idx=None,
                    )
                    self.date_time_segments.append(segment)

                    prev_date = date
                    prev_time = time

    def _finalize_dt_segments(self) -> None:
        # get the start and end segments
        for i in range(len(self.date_time_segments) - 1):
            seg_1, seg_2 = self.date_time_segments[i], self.date_time_segments[i + 1]
            start_idx = self.calculate_index_from_cumulative_counts(
                seg_1.start_offset, self.cum_total_counts_np
            )
            end_idx = self.calculate_index_from_cumulative_counts(
                seg_2.start_offset - 1, self.cum_total_counts_np
            )

            seg_1.end_offset = seg_2.start_offset
            seg_1.start_idx = start_idx
            seg_1.end_idx = end_idx

        final_segment = self.date_time_segments[-1]
        final_segment.start_idx = self.calculate_index_from_cumulative_counts(
            final_segment.start_offset, self.cum_total_counts_np
        )
        final_segment.end_idx = len(self.cum_total_counts)
        final_segment.end_offset = self.total_num_rows

    def __len__(self) -> int:
        return len(self.date_time_segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.stack(
            [self._get_dt_sample(i) for i in range(max(0, idx - self.time_context_length), idx + 1)]
        )

    def _get_dt_sample(self, idx: int) -> torch.Tensor:
        seg = self.date_time_segments[idx]
        df_list: List[torch.Tensor] = []
        for total_idx in range(seg.start_idx, seg.end_idx + 1):
            group: RowGroupOffset = self.cum_total_counts[total_idx]
            self._load_pq_file(
                file_idx=group.file_idx,
                row_group_idx=group.row_group_idx,
            )

            # start offset within this group is the total start offset minus the group start offset.
            # If it's greater than zero, we need to discard some rows from the start of this row
            # group.
            start_offset = max(seg.start_offset - group.offset, 0)

            # end offset within this group is the smaller of the total end row - the group start row
            # and the size of the group. Basically if the end offset is larger than row group, just
            # take the whole group. Otherwise we'll discard some rows from the end of this row
            # group.
            end_offset = min(
                seg.end_offset - group.offset,
                self.file_row_group_counts[group.file_idx][group.row_group_idx],
            )

            # The start row offset is ahead of the starting point for this row group.
            # Need to select the rows starting at the offset.
            df_list.append(self.pq_cache.df.to_torch()[start_offset:end_offset])

        return torch.vstack(df_list)
