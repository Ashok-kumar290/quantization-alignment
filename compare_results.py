#!/usr/bin/env python3
"""
Cross-precision comparison script.

Runs the full experiment pipeline across multiple quantization levels
for the same model, then generates comparative analysis.

Usage:
    python compare_results.py --model mistral --precisions fp16 8bit 4bit
    python compare_results.py --model llama3 --precisions fp16 8bit 4bit 3bit
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

from models.model_loader import ModelLoader
from models.activation_collector import ActivationCollector
from datasets.sycophancy_dataset import SycophancyDataset
from datasets.truthfulness_dataset import TruthfulnessDataset
from datasets.safety_dataset import SafetyDataset
from experiments.evaluator import AlignmentEvaluator
from probes.linear_probe import AlignmentProbe, LayerwiseProbeAnalysis
from analysis.geometric import GeometricAnalyzer
from utils.visualization import ResultsVisualizer
from utils.metrics import compute_metrics, format_metrics_table
from run_experiment import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_precision(
    model_name: str,
    precision: str,
    n_sycophancy: int = 50,
    n_truthfulness: int = 25,
    n_safety: int = 34,
) -> Dict[str, Any]:
    """Run evaluation at a single precision level and return results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {model_name} @ {precision}")
    logger.info(f"{'='*60}")

    config = ExperimentConfig(model=model_name, precision=precision)

    # Load model
    loader = ModelLoader(
        model_name=config.get_model_path(),
        quantization_config=config.get_quantization_config(),
    )
    model, tokenizer = loader.load()

    results = {"precision": precision}

    # Evaluate behaviors
    evaluator = AlignmentEvaluator(model=model, tokenizer=tokenizer)

    syc_ds = SycophancyDataset(n_samples=n_sycophancy)
    results["sycophancy"] = evaluator.evaluate_sycophancy(syc_ds)

    truth_ds = TruthfulnessDataset(n_samples=n_truthfulness)
    results["truthfulness"] = evaluator.evaluate_truthfulness(truth_ds)

    safety_ds = SafetyDataset(n_samples=n_safety)
    results["safety"] = evaluator.evaluate_safety(safety_ds)

    # Collect per-layer activations for probing
    collector = ActivationCollector(model, tokenizer)
    syc_activations = collector.collect_all_layers(syc_ds)
    results["activations_per_layer"] = syc_activations["per_layer"]
    results["activation_labels"] = syc_activations["labels"]

    # Train per-layer probes
    layerwise = LayerwiseProbeAnalysis()
    probe_results = layerwise.train_all_layers(
        syc_activations["per_layer"],
        syc_activations["labels"],
    )
    results["probe_per_layer"] = probe_results
    results["direction_vectors"] = layerwise.get_direction_vectors()

    # Compute summary
    results["summary"] = compute_metrics(results)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def compare_precisions(
    all_results: Dict[str, Dict],
    output_dir: Path,
):
    """Generate cross-precision comparison analysis and plots."""
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Precision Comparison Analysis")
    logger.info("=" * 60)

    precisions = sorted(all_results.keys())
    reference = precisions[0]  # Typically fp16

    # 1. Alignment direction stability across precisions
    # Compare the direction vectors from the best probing layer
    best_directions = {}
    for prec, res in all_results.items():
        directions = res.get("direction_vectors", {})
        if directions:
            # Use the direction from the layer with highest probe accuracy
            probe_results = res.get("probe_per_layer", {})
            if probe_results:
                best_layer = max(probe_results, key=lambda k: probe_results[k]["accuracy"])
                if best_layer in directions:
                    best_directions[prec] = directions[best_layer]

    direction_stability = {}
    sim_matrix = None
    sim_labels = None
    if len(best_directions) > 1:
        direction_stability = GeometricAnalyzer.alignment_vector_stability(
            best_directions, reference_precision=reference
        )
        sim_matrix, sim_labels = GeometricAnalyzer.cosine_similarity_matrix(best_directions)
        logger.info(f"Direction stability: {direction_stability}")

    # 2. Per-layer CKA comparison
    layer_sensitivity_results = {}
    if reference in all_results:
        ref_acts = all_results[reference].get("activations_per_layer", {})
        for prec in precisions:
            if prec == reference:
                continue
            quant_acts = all_results[prec].get("activations_per_layer", {})
            if ref_acts and quant_acts:
                # Convert torch tensors to numpy
                ref_np = {k: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
                          for k, v in ref_acts.items()}
                quant_np = {k: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
                            for k, v in quant_acts.items()}
                sensitivity = GeometricAnalyzer.layer_sensitivity(ref_np, quant_np)
                layer_sensitivity_results[prec] = sensitivity

    # 3. CKA between full activation matrices (using best layer)
    cka_by_precision = {}
    for prec, res in all_results.items():
        acts = res.get("activations_per_layer", {})
        if acts:
            probe_results = res.get("probe_per_layer", {})
            if probe_results:
                best_layer = max(probe_results, key=lambda k: probe_results[k]["accuracy"])
                if best_layer in acts:
                    a = acts[best_layer]
                    cka_by_precision[prec] = a.cpu().numpy() if hasattr(a, 'cpu') else np.asarray(a)

    cka_matrix = None
    cka_labels = None
    if len(cka_by_precision) > 1:
        cka_matrix, cka_labels = GeometricAnalyzer.representational_similarity_matrix(cka_by_precision)
        logger.info(f"CKA matrix labels: {cka_labels}")

    # 4. Generate comparison visualizations
    comparison_results = {
        "precision_results": {p: r for p, r in all_results.items()},
        "geometric": {
            "direction_stability": direction_stability,
            "direction_similarity_matrix": sim_matrix.tolist() if sim_matrix is not None else None,
            "direction_similarity_labels": sim_labels,
            "layer_sensitivity": layer_sensitivity_results,
        },
    }

    viz = ResultsVisualizer(comparison_results, output_dir / "figures")

    # Behavior comparison plot
    viz.plot_sycophancy_by_quantization(
        precision_results={p: r for p, r in all_results.items()}
    )

    # Alignment direction heatmap
    if sim_matrix is not None:
        viz.plot_alignment_heatmap(sim_matrix, sim_labels)

    # Layer sensitivity plots
    for prec, sensitivity in layer_sensitivity_results.items():
        viz.plot_layer_sensitivity(
            layer_metrics=sensitivity,
            metric_name="mean_cosine_sim",
        )

    # CKA matrix
    if cka_matrix is not None:
        viz.plot_cka_matrix(cka_matrix, cka_labels)

    # 5. Save comparison results
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_serializable(i) for i in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    summary = {
        "precisions": precisions,
        "direction_stability": _to_serializable(direction_stability),
        "behavior_comparison": {},
    }

    for prec, res in all_results.items():
        summary["behavior_comparison"][prec] = {
            "sycophancy_rate": res.get("sycophancy", {}).get("sycophancy_rate"),
            "truthfulness_score": res.get("truthfulness", {}).get("truthfulness_score"),
            "refusal_rate": res.get("safety", {}).get("refusal_rate"),
        }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(_to_serializable(summary), f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Precision':<12} {'Sycophancy':>12} {'Truthfulness':>14} {'Refusal':>10} {'Direction Sim':>15}")
    print("-" * 70)
    for prec in precisions:
        syc = all_results[prec].get("sycophancy", {}).get("sycophancy_rate", 0)
        truth = all_results[prec].get("truthfulness", {}).get("truthfulness_score", 0)
        ref = all_results[prec].get("safety", {}).get("refusal_rate", 0)
        dir_sim = direction_stability.get(prec, 0)
        print(f"{prec:<12} {syc:>12.4f} {truth:>14.4f} {ref:>10.4f} {dir_sim:>15.4f}")
    print("=" * 70)

    logger.info(f"Comparison results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cross-precision comparison")
    parser.add_argument("--model", type=str, default="mistral",
                        choices=["llama3", "mistral", "qwen"])
    parser.add_argument("--precisions", nargs="+", default=["fp16", "8bit", "4bit"],
                        help="Precision levels to compare")
    parser.add_argument("--n-sycophancy", type=int, default=50)
    parser.add_argument("--n-truthfulness", type=int, default=25)
    parser.add_argument("--n-safety", type=int, default=34)
    parser.add_argument("--output-dir", type=str, default="results/comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for precision in args.precisions:
        all_results[precision] = run_single_precision(
            model_name=args.model,
            precision=precision,
            n_sycophancy=args.n_sycophancy,
            n_truthfulness=args.n_truthfulness,
            n_safety=args.n_safety,
        )

    compare_precisions(all_results, output_dir)


if __name__ == "__main__":
    main()
