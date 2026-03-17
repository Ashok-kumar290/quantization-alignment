"""
Visualization utilities for experiment results.

Generates publication-quality plots for:
- Behavior metrics across quantization levels
- Alignment vector similarity heatmaps
- Layer sensitivity profiles
- Intervention effect grids
- PCA projections of hidden states
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

logger = logging.getLogger(__name__)

# Use a clean style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


class ResultsVisualizer:
    """Generates all experiment plots."""

    def __init__(self, results: Dict[str, Any], output_dir: Path):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig: plt.Figure, name: str):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot: {path}")

    # ------------------------------------------------------------------
    # Behavioral metrics across quantization levels
    # ------------------------------------------------------------------

    def plot_sycophancy_by_quantization(
        self,
        precision_results: Optional[Dict[str, Dict]] = None,
    ):
        """
        Bar chart of sycophancy rate at each quantization level.

        Args:
            precision_results: Dict mapping precision label -> result dict
                that includes 'sycophancy' -> {'sycophancy_rate': float}.
                If None, uses self.results (single-run mode).
        """
        if precision_results is None:
            # Single-run mode: plot category breakdown
            syc = self.results.get("sycophancy", {})
            cat_rates = syc.get("category_rates", {})
            if not cat_rates:
                logger.warning("No sycophancy category data to plot")
                return

            fig, ax = plt.subplots()
            cats = sorted(cat_rates.keys())
            rates = [cat_rates[c] for c in cats]
            bars = ax.bar(cats, rates, color=sns.color_palette("Set2", len(cats)))
            ax.set_ylabel("Sycophancy Rate")
            ax.set_title("Sycophancy Rate by Category")
            ax.set_ylim(0, 1)
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
            plt.xticks(rotation=45, ha="right")
            self._save_fig(fig, "sycophancy_by_category")
            return

        # Multi-precision mode
        precisions = sorted(precision_results.keys())
        syc_rates = []
        truth_scores = []
        refusal_rates = []

        for prec in precisions:
            r = precision_results[prec]
            syc_rates.append(r.get("sycophancy", {}).get("sycophancy_rate", 0))
            truth_scores.append(r.get("truthfulness", {}).get("truthfulness_score", 0))
            refusal_rates.append(r.get("safety", {}).get("refusal_rate", 0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sycophancy
        colors_syc = sns.color_palette("Reds_d", len(precisions))
        axes[0].bar(precisions, syc_rates, color=colors_syc)
        axes[0].set_title("Sycophancy Rate")
        axes[0].set_ylabel("Rate")
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(syc_rates):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        # Truthfulness
        colors_truth = sns.color_palette("Greens_d", len(precisions))
        axes[1].bar(precisions, truth_scores, color=colors_truth)
        axes[1].set_title("Truthfulness Score")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(truth_scores):
            axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        # Refusal
        colors_ref = sns.color_palette("Blues_d", len(precisions))
        axes[2].bar(precisions, refusal_rates, color=colors_ref)
        axes[2].set_title("Refusal Rate (Harmful)")
        axes[2].set_ylabel("Rate")
        axes[2].set_ylim(0, 1)
        for i, v in enumerate(refusal_rates):
            axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        fig.suptitle("Alignment Metrics vs Quantization Level", fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, "alignment_vs_quantization")

    # ------------------------------------------------------------------
    # Alignment vector similarity heatmap
    # ------------------------------------------------------------------

    def plot_alignment_heatmap(
        self,
        similarity_matrix: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        title: str = "Alignment Direction Cosine Similarity",
    ):
        """
        Heatmap of cosine similarity between alignment direction vectors
        across quantization levels.
        """
        if similarity_matrix is None:
            # Try to extract from results
            geo = self.results.get("geometric", {})
            similarity_matrix = geo.get("direction_similarity_matrix")
            labels = geo.get("direction_similarity_labels")
            if similarity_matrix is None:
                logger.warning("No alignment similarity data to plot")
                return
            similarity_matrix = np.array(similarity_matrix)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(similarity_matrix, cmap="RdYlGn", vmin=-1, vmax=1)

        # Annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{similarity_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(similarity_matrix[i, j]) > 0.7 else "black")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Cosine Similarity")
        self._save_fig(fig, "alignment_direction_heatmap")

    # ------------------------------------------------------------------
    # Layer sensitivity
    # ------------------------------------------------------------------

    def plot_layer_sensitivity(
        self,
        layer_metrics: Optional[Dict[str, Dict]] = None,
        metric_name: str = "mean_cosine_sim",
    ):
        """
        Line plot showing per-layer sensitivity to quantization.

        Args:
            layer_metrics: Dict of layer_name -> metric dict from
                GeometricAnalyzer.layer_sensitivity().
            metric_name: Which metric to plot.
        """
        if layer_metrics is None:
            layer_metrics = self.results.get("geometric", {}).get("layer_sensitivity", {})
            if not layer_metrics:
                logger.warning("No layer sensitivity data to plot")
                return

        # Sort by layer index
        def _layer_num(name: str) -> int:
            for part in reversed(name.split(".")):
                if part.isdigit():
                    return int(part)
            return 0

        sorted_layers = sorted(layer_metrics.keys(), key=_layer_num)
        layer_indices = [_layer_num(l) for l in sorted_layers]
        values = [layer_metrics[l][metric_name] for l in sorted_layers]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(layer_indices, values, "o-", color="#2196F3", linewidth=2, markersize=5)
        ax.fill_between(layer_indices, values, alpha=0.15, color="#2196F3")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title("Layer Sensitivity to Quantization")
        self._save_fig(fig, "layer_sensitivity")

    # ------------------------------------------------------------------
    # Probe accuracy per layer
    # ------------------------------------------------------------------

    def plot_probe_accuracy_per_layer(
        self,
        probe_results: Dict[str, Dict[str, float]],
        title: str = "Probe Accuracy by Layer",
    ):
        """
        Line plot of probe accuracy at each layer.

        Args:
            probe_results: Dict of layer_name -> {'accuracy': float, ...}.
        """
        def _layer_num(name: str) -> int:
            for part in reversed(name.split(".")):
                if part.isdigit():
                    return int(part)
            return 0

        sorted_layers = sorted(probe_results.keys(), key=_layer_num)
        layer_indices = [_layer_num(l) for l in sorted_layers]
        accuracies = [probe_results[l]["accuracy"] for l in sorted_layers]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(layer_indices, accuracies, "s-", color="#E91E63", linewidth=2, markersize=6)
        ax.axhline(y=0.5, color="gray", linestyle="--", label="Chance level")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(title)
        ax.set_ylim(0.4, 1.05)
        ax.legend()
        self._save_fig(fig, "probe_accuracy_per_layer")

    # ------------------------------------------------------------------
    # Intervention effect heatmap
    # ------------------------------------------------------------------

    def plot_intervention_effects(
        self,
        intervention_results: Optional[Dict] = None,
    ):
        """
        Heatmap showing behavior metric as a function of layer and alpha.
        """
        if intervention_results is None:
            intervention_results = self.results.get("interventions", {})
            if not intervention_results:
                logger.warning("No intervention data to plot")
                return

        for behavior_name, behavior_data in intervention_results.items():
            layers = sorted(behavior_data.keys())
            if not layers:
                continue

            alphas = sorted(behavior_data[layers[0]].keys())
            grid = np.zeros((len(layers), len(alphas)))

            for i, layer in enumerate(layers):
                for j, alpha in enumerate(alphas):
                    grid[i, j] = behavior_data[layer][alpha]["metric"]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r")

            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:.1f}" for a in alphas])
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"L{l}" for l in layers])
            ax.set_xlabel("Steering Strength (alpha)")
            ax.set_ylabel("Layer")
            ax.set_title(f"Intervention Effect: {behavior_name}")
            fig.colorbar(im, ax=ax, label="Behavior Metric")

            # Annotations
            for i in range(len(layers)):
                for j in range(len(alphas)):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)

            self._save_fig(fig, f"intervention_{behavior_name}")

    # ------------------------------------------------------------------
    # PCA projection
    # ------------------------------------------------------------------

    def plot_pca_projections(
        self,
        pca_result: Dict[str, Any],
        title: str = "PCA of Hidden States",
    ):
        """
        Scatter plot of PCA-projected hidden states, colored by label.
        """
        projections = pca_result["projections"]
        labels = pca_result.get("labels")
        evr = pca_result.get("explained_variance_ratio", [0, 0])

        fig, ax = plt.subplots(figsize=(8, 6))

        if labels is not None:
            labels = np.asarray(labels)
            unique_labels = np.unique(labels)
            colors = sns.color_palette("Set1", len(unique_labels))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    projections[mask, 0], projections[mask, 1],
                    c=[colors[i]], alpha=0.6, s=30, label=f"Class {label}"
                )
            ax.legend()
        else:
            ax.scatter(projections[:, 0], projections[:, 1], alpha=0.6, s=30)

        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        ax.set_title(title)
        self._save_fig(fig, "pca_projections")

    # ------------------------------------------------------------------
    # Multi-precision PCA comparison
    # ------------------------------------------------------------------

    def plot_pca_comparison(
        self,
        pca_results_by_precision: Dict[str, Dict[str, Any]],
    ):
        """
        Side-by-side PCA projections for different precision levels.
        """
        precisions = sorted(pca_results_by_precision.keys())
        n = len(precisions)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, prec in zip(axes, precisions):
            pca_result = pca_results_by_precision[prec]
            projections = pca_result["projections"]
            labels = pca_result.get("labels")
            evr = pca_result.get("explained_variance_ratio", [0, 0])

            if labels is not None:
                labels = np.asarray(labels)
                unique_labels = np.unique(labels)
                colors = sns.color_palette("Set1", len(unique_labels))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(
                        projections[mask, 0], projections[mask, 1],
                        c=[colors[i]], alpha=0.6, s=20, label=f"Class {label}"
                    )
                ax.legend(fontsize=8)
            else:
                ax.scatter(projections[:, 0], projections[:, 1], alpha=0.6, s=20)

            ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
            ax.set_title(f"{prec}")

        fig.suptitle("PCA Projections Across Quantization Levels", fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, "pca_comparison")

    # ------------------------------------------------------------------
    # Representational similarity matrix (CKA)
    # ------------------------------------------------------------------

    def plot_cka_matrix(
        self,
        cka_matrix: np.ndarray,
        labels: List[str],
        title: str = "Representational Similarity (CKA)",
    ):
        """Heatmap of CKA similarity between precision levels."""
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cka_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=10)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="CKA")
        self._save_fig(fig, "cka_similarity_matrix")

    # ------------------------------------------------------------------
    # Summary dashboard
    # ------------------------------------------------------------------

    def generate_all_plots(self):
        """Generate all available plots from the results dict."""
        if "sycophancy" in self.results:
            self.plot_sycophancy_by_quantization()

        if "interventions" in self.results:
            self.plot_intervention_effects()

        geo = self.results.get("geometric", {})
        if "layer_sensitivity" in geo:
            self.plot_layer_sensitivity()
        if "direction_similarity_matrix" in geo:
            self.plot_alignment_heatmap()

        probe = self.results.get("probe_sycophancy_per_layer", {})
        if probe:
            self.plot_probe_accuracy_per_layer(probe, "Sycophancy Probe by Layer")

        logger.info(f"All plots saved to {self.output_dir}")
