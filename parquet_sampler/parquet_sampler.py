"""Custom Pytorch samplers for shuffling a parquet dataset.
TODO: make this inherit from the RandomSampler or otherwise.
"""

from typing import Iterator
import torch
from torch.utils.data import Sampler

from parquet_dataset.parquet_dataset import ParquetDataset, RowGroupOffset


class ParquetBatchedSampler(Sampler[int]):
    """Samples elements randomly while batching by Parquet file to minimize disk
    i/o. Randomly orders the N parquet files and then provides random indices into
    the rows of each parquet file, ensuring that each file is fully sampled
    before movign on the to next.

    Args:
        data_source (Dataset): dataset to sample from generator (Generator):
        Generator used in sampling.
    """

    data_source: ParquetDataset

    def __init__(
        self,
        data_source: ParquetDataset,
        generator=None,
    ) -> None:
        super().__init__()
        self.data_source = data_source
        self.generator = generator

        self.n_groups: int = len(self.data_source.cum_total_counts)

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset. No support for subsampling or variable size."""
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Randomly sample from an ordering or partitions, shuffling all of the samples within a
        # partition.
        for i in torch.randperm(self.n_groups, generator=generator):
            group: RowGroupOffset = self.data_source.cum_total_counts[i]
            yield from group.offset + torch.randperm(
                self.data_source.file_row_group_counts[group.file_idx][group.row_group_idx]
            )

    def __len__(self) -> int:
        return len(self.data_source)
