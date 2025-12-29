"""
Visualization utilities for evaluation metrics (GT-based).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SOURCE_NAMES = ["drums", "bass", "other", "vocals"]


def plot_metric_comparison(metrics_data, metric_name, save_path=None):
    scenarios = list(metrics_data.keys())

    data = {
        scenario: [metrics_data[scenario][s][metric_name] for s in SOURCE_NAMES]
        for scenario in scenarios
    }

    x = np.arange(len(SOURCE_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["lightgray", "salmon", "lightgreen"]
    labels = ["GT vs Baseline", "GT vs Attack", "GT vs Defense"]

    for i, (scenario, label, color) in enumerate(zip(scenarios, labels, colors)):
        bars = ax.bar(x + (i - 1) * width, data[scenario], width,
                      label=label, color=color)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SOURCE_NAMES])
    ax.set_ylabel(f"{metric_name} (dB)")
    ax.set_title(f"{metric_name} Comparison")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    return fig


def plot_all_metrics(metrics_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["SDR", "SIR", "SAR"]:
        plot_metric_comparison(
            metrics_data,
            metric,
            output_dir / f"{metric}_comparison.png"
        )


def create_summary_plot(metrics_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["SDR", "SIR", "SAR"]
    scenarios = list(metrics_data.keys())

    colors = ["lightgray", "salmon", "lightgreen"]
    labels = ["GT vs Baseline", "GT vs Attack", "GT vs Defense"]

    for ax, metric in zip(axes, metrics):
        data = {
            scenario: [metrics_data[scenario][s][metric] for s in SOURCE_NAMES]
            for scenario in scenarios
        }

        x = np.arange(len(SOURCE_NAMES))
        width = 0.25

        for i, (scenario, label, color) in enumerate(zip(scenarios, labels, colors)):
            ax.bar(x + (i - 1) * width, data[scenario], width,
                   label=label, color=color)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in SOURCE_NAMES])
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[0].legend()
    plt.suptitle("Demucs Attack & Defense Evaluation (GT Reference)")
    plt.tight_layout()

    save_path = output_dir / "summary_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return str(save_path)
