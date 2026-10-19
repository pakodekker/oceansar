"""Shared save/show lifecycle for processing diagnostics."""
from pathlib import Path

from matplotlib import pyplot as plt


def finish_plot(figure, path, *, save, show=False, **save_kwargs):
    """Save if requested, display if requested or unsaved, and close the figure.

    path is a destination filename, used only when save is True. Explicit show
    allows notebooks to display figures while also saving them to disk.
    """
    try:
        if save:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, **save_kwargs)
        if show or not save:
            plt.show()
    finally:
        plt.close(figure)
