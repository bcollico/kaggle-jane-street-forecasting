"Parquet dataset sampler unittest."

import unittest

import numpy as np

from parquet_sampler.parquet_sampler import ParquetBatchedSampler
from parquet_dataset.parquet_dataset import ParquetDataset


class MockParquetDataSet:
    """Simple dataset mock for the parquet dataset to test the sampler."""

    def __init__(self):
        self.pq_dataset = ParquetDataset(file_paths=[], logging=False)

        # Mock ParquetDataset with 3 "files" and differing numbers of row groups per file
        # Each "file" is a list of the number of rows in each row group.
        self.pq_dataset.file_row_group_counts = [[10, 90, 900], [1000, 8000], [1000]]
        self.pq_dataset.file_row_counts = [sum(c) for c in self.pq_dataset.file_row_group_counts]

        # Get the cumulative counts from the file and row group counts.
        self.pq_dataset._setup_cumulative_counts()

        self.num_samples = sum(self.pq_dataset.file_row_counts)


class TestParquetSampler(unittest.TestCase):
    """Unit test fixture for the ParquetBatchedSampler"""

    def setUp(self):
        self.mock_dataset = MockParquetDataSet()
        self.pq_sampler = ParquetBatchedSampler(data_source=self.mock_dataset.pq_dataset)

    def test_should_fully_sample_data(self):
        """Check that we fully sample all indices expected."""
        # precondition: create a set of all indices to be sampled
        sample_set: set[int] = set(range(self.mock_dataset.num_samples))

        # under test: fully sample and remove from the set on each sample
        for sample in self.pq_sampler:
            sample_set.remove(sample.item())

        # postcondition: sample_set should be empty
        self.assertEqual(sample_set, set())


if __name__ == "__main__":
    unittest.main()
