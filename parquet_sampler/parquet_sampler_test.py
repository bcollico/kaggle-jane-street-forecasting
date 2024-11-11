"Parquet dataset sampler unittest."

import unittest

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

    def test_should_sample_in_batch_order(self):
        """Check that we fully sample all indices expected."""
        # precondition: create a set of all indices to be sampled
        sample_list: list[int] = []

        # under test: Append the order in which row groups are sampled from.
        for sample in self.pq_sampler:
            # Get the rowgroup/file total index.
            idx = self.mock_dataset.pq_dataset.calculate_index_from_cumulative_counts(
                sample.item(), self.mock_dataset.pq_dataset.cum_total_counts_np
            )

            # Add new rowgroup/file indices to the list if it's different than the latest.
            if not sample_list or idx != sample_list[-1]:
                sample_list.append(idx)

        # postcondition: The unique list of row group orderings should be equivalent to the
        # set of all row groups.
        sample_set: set[int] = set(sample_list)
        self.assertEqual(sample_set, set(range(len(self.mock_dataset.pq_dataset.cum_total_counts))))

        # postcondition: The unique list of row group orderings should be the same size as the
        # non-unique ordered set of row groups. We should be sampling row groups in order,
        # so we should only ever add it to sample_list once.
        self.assertEqual(len(sample_set), len(sample_list))


if __name__ == "__main__":
    unittest.main()
