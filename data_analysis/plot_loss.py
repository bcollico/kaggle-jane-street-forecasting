import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

K_LOSS_DIRECTORY = Path("ckpt/")
import shutil

def plot(subfolder_name: str) -> None:
    """Plot the training and validation losses for the rq"""

    # Sample data generation (replace these with your actual data)
    responder_mae_train = torch.empty((0, 9), requires_grad=False)
    responder_mae_val = torch.empty((0, 9), requires_grad=False)
    loss_train = torch.empty((0, 9), requires_grad=False)
    loss_val = torch.empty((0, 9), requires_grad=False)

    for checkpoint in sorted(
        (K_LOSS_DIRECTORY / subfolder_name).glob("*.pt"),
        key=lambda x: int(x.stem) if "meta" not in x.stem else 0,
    ):
        print(checkpoint)

        if "meta" in checkpoint.stem:
            continue

        ckpt = torch.load(checkpoint)

        responder_mae_train = torch.vstack(
            (responder_mae_train, ckpt["training"]["mean_responder_error"].detach().cpu())
        )
        responder_mae_val = torch.vstack(
            (responder_mae_val, ckpt["validation"]["mean_responder_error"].detach().cpu())
        )

        loss_train = torch.vstack((loss_train, ckpt["training"]["mean_loss_vector"].cpu()))
        loss_val = torch.vstack((loss_val, ckpt["validation"]["mean_loss_vector"].cpu()))

    responder_mae_train = torch.hstack(
        (responder_mae_train, responder_mae_train.sum(1, keepdim=True))
    ).detach()
    responder_mae_val = torch.hstack(
        (responder_mae_val, responder_mae_val.sum(1, keepdim=True))
    ).detach()

    loss_train = torch.hstack((loss_train, loss_train.sum(1, keepdim=True))).detach()
    loss_val = torch.hstack((loss_val, loss_val.sum(1, keepdim=True))).detach()

    timesteps = np.arange(responder_mae_train.shape[0])

    # Create two figures, each with a 2x5 grid of subplots
    fig1, axes1 = plt.subplots(2, 5, figsize=(15, 8), sharex=True, sharey=False)
    fig2, axes2 = plt.subplots(2, 5, figsize=(15, 8), sharex=True, sharey=False)

    # Plot Responder MAE
    for row in range(2):
        for col in range(5):  # Iterate over the 10 scalars
            ax = axes1[row, col]
            scalar_idx = row * 5 + col

            # Plot train and val time-series for the current scalar
            ax.plot(timesteps, responder_mae_train[:, scalar_idx], label="Train", marker="o")
            ax.plot(timesteps, responder_mae_val[:, scalar_idx], label="Val", marker="x")
            ax.set_title(f"Responder MAE Scalar {scalar_idx}")

            # Add legend only for the first plot
            if row == 0 and col == 0:
                ax.legend()

    fig1.text(0.5, 0.04, "Timesteps", ha="center", fontsize=14)
    fig1.text(0.04, 0.5, "Metric Value", va="center", rotation="vertical", fontsize=14)
    fig1.suptitle("Responder MAE Over Time", fontsize=16)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Plot Loss
    for row in range(2):
        for col in range(5):  # Iterate over the 10 scalars
            ax = axes2[row, col]
            scalar_idx = row * 5 + col

            # Plot train and val time-series for the current scalar
            ax.plot(timesteps, loss_train[:, scalar_idx], label="Train", marker="o")
            ax.plot(timesteps, loss_val[:, scalar_idx], label="Val", marker="x")
            ax.set_title(f"Loss Scalar {scalar_idx}")

            # Add legend only for the first plot
            if row == 0 and col == 0:
                ax.legend()

    fig2.text(0.5, 0.04, "Timesteps", ha="center", fontsize=14)
    fig2.text(0.04, 0.5, "Metric Value", va="center", rotation="vertical", fontsize=14)
    fig2.suptitle("Loss Over Time", fontsize=16)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Display the plots
    plt.show()


if __name__ == "__main__":
    # plot("1734587441.0320776")
    # plot("1734673544.8482938")
    # plot("1735111914.702341")
    # plot("1735235606.2772698")
    plot("1735274590.9468231")