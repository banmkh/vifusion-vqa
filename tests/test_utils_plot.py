from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from src.utils.plot import plot_training_curves


def test_plot_training_curves_runs():
    plot_training_curves([1.0, 0.8, 0.6], [1.2, 1.1, 1.0])
