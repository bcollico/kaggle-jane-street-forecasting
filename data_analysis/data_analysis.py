"""Script to generate plots analyzing features and responder distributions in the dataset"""

from typing import Callable, Optional

from pathlib import Path

import polars as pl
import pyarrow
import pyarrow.dataset
import pyarrow.parquet
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

K_OUT_PATH = Path("./data_analysis/plots/")
K_CHUNK_SIZE = 1024


def plot(
    array: np.ndarray, n_bins: int, col_name: str, output_folder=None, title_suffix: str = ""
) -> None:
    plt.hist(array, bins=n_bins, edgecolor="black")

    # Add labels and title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(col_name + title_suffix)

    # Save the histogram to a file
    path_out = K_OUT_PATH if output_folder is None else K_OUT_PATH / output_folder
    path_out.mkdir(parents=True, exist_ok=True)
    output_file = path_out / col_name  # Specify the file name and format
    plt.savefig(output_file, dpi=300)  # Save at 300 DPI for high-quality output
    plt.clf()


def create_feature_histograms(
    func: Callable[[int, Path, int, bool], None],
    output_folder: str = None,
    n_bins: int = 100,
) -> None:
    """Create histograms of the distribution over feature values for each feature. Optionally pass
    a folder name to output the images to. Optionally provide a lambda for transforming the input
    features before creating the histogram."""

    for i in range(79):
        func(i=i, output_folder=output_folder, n_bins=n_bins, is_feature=True)

    for i in range(9):
        func(i=i, output_folder=output_folder, n_bins=n_bins, is_feature=False)


def make_dataset() -> pyarrow.parquet.ParquetDataset:
    """Make a pyarrow dataset for all of the input parquet files."""
    return pyarrow.parquet.ParquetDataset(
        "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/"
    )


def get_col_name(i: int, is_feature: bool) -> str:
    """Get the column name for feature or responder i."""
    return f"feature_{str(i).zfill(2)}" if is_feature else "responder_{str(i).zfill(1)}"


def get_col_samples(
    col_name: str, samples: int, dataset: Optional[pyarrow.parquet.ParquetDataset] = None
) -> np.ndarray:
    """Get N samples from a column in the overall parquet dataset."""

    if not dataset:
        dataset = make_dataset()

    feature_i_col: pl.DataFrame = pl.from_arrow(make_dataset().read(columns=[col_name]))

    # (N,) numpy array for this column
    col_array = feature_i_col.to_numpy()

    # Remove NaN from dataset
    # TODO try converting NaN to zero or other strategies e.g. Nearest Neighbor on the last
    # timestep
    col_array = col_array[np.logical_not(np.isnan(col_array))]

    return np.random.choice(col_array, size=samples)


# def run_data_nf(
#     i: int,
#     output_folder: str = None,
#     n_bins: int = 100,
#     is_feature: bool = True,
#     samples: int = 100000,
# ):
#     """Create a transform using normalizing flows to a 1D gaussian"""
#     import torch
#     import normflows as nf
#     import tqdm
#     col_name = get_col_name(i, is_feature=is_feature)
#     print(col_name)

#     col_array: torch.Tensor = (
#         torch.from_numpy(get_col_samples(col_name=col_name, samples=samples))
#         .to("cuda")
#         .view(-1, 1)
#     )

#     # Set up model

#     # Define 2D Gaussian base distribution
#     base = nf.distributions.base.DiagGaussian(1)

#     # Define list of flows
#     num_layers = 32
#     flows = []
#     for _ in range(num_layers):
#         # Neural network with two hidden layers having 64 units each
#         # Last layer is initialized by zeros making training more stable
#         param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
#         # Add flow layer
#         flows.append(nf.flows.AffineCouplingBlock(param_map))
#         # Swap dimensions
#         flows.append(nf.flows.Permute(2, mode="swap"))

#     # Construct flow model
#     model = nf.NormalizingFlow(base, flows).to("cuda")

#     max_iter = 4000
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

#     for _ in tqdm.tqdm(range(max_iter)):
#         optimizer.zero_grad()

#         inp = col_array.expand(-1,2)
#         loss = model.forward_kld(inp)

#         if ~(torch.isnan(loss) | torch.isinf(loss)):
#             loss.backward()
#             optimizer.step()

#     import pdb; pdb.set_trace()
#     model.eval()
#     inverse_samples: torch.Tensor = model.inverse(
#         torch.from_numpy(get_col_samples(col_name=col_name, samples=samples))
#         .to("cuda")
#         .view(-1, 1).expand(-1, 2)
#     )

#     plot(
#         array=inverse_samples.cpu().numpy(),
#         n_bins=n_bins,
#         col_name=col_name,
#         output_folder=output_folder,
#     )


def run_data_yj(
    i: int,
    output_folder: str = None,
    n_bins: int = 100,
    is_feature: bool = True,
    samples: int = 100000,
):
    """Run with Yeo-Johnson transform to normalize the data."""
    col_name = get_col_name(i, is_feature=is_feature)
    print(col_name)

    col_array = get_col_samples(col_name=col_name, samples=samples)
    col_array, max_lambda = stats.yeojohnson(col_array)

    out_path = K_OUT_PATH if output_folder is None else K_OUT_PATH / output_folder
    out_path.mkdir(parents=True, exist_ok=True)
    with open((out_path / col_name).with_suffix(".txt"), "w") as f:
        f.write(f"{max_lambda:0.10f}")

    plot(
        array=col_array,
        n_bins=n_bins,
        col_name=col_name,
        output_folder=output_folder,
        title_suffix=f"_{max_lambda:0.4f}",
    )


if __name__ == "__main__":
    create_feature_histograms(func=run_data_yj, output_folder="yj3")
