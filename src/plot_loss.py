import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, log_path, logger=None):
    """Draw and save per-snapshot training loss curves."""
    if not loss_history:
        if logger is not None:
            logger.info("No loss history found, skip plotting loss curve.")
        return

    os.makedirs(log_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plotted = False
    for snapshot_id, losses in loss_history.items():
        if len(losses) == 0:
            continue
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, label=f"Snapshot {snapshot_id}")
        plotted = True

    if not plotted:
        if logger is not None:
            logger.info("Loss history is empty after filtering, skip plotting loss curve.")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_fig_path = os.path.join(log_path, "loss_curve.png")
    plt.savefig(loss_fig_path, dpi=200)
    plt.close()

    if logger is not None:
        logger.info(f"Loss curve saved to: {loss_fig_path}")
