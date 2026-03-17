"""
Aggregate metric computation for experiment results.

Computes and organizes all behavior and representation metrics
into a summary structure suitable for reporting.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary metrics from raw experiment results.

    Args:
        results: Dict containing raw results from each pipeline stage:
            - 'sycophancy': output of evaluate_sycophancy()
            - 'truthfulness': output of evaluate_truthfulness()
            - 'safety': output of evaluate_safety()
            - 'probe_sycophancy': probe evaluation metrics
            - 'probe_safety': probe evaluation metrics
            - 'interventions': intervention sweep results
            - 'geometric': geometric analysis results

    Returns:
        Summary dict with organized metrics.
    """
    summary = {
        "behavior_metrics": {},
        "representation_metrics": {},
        "intervention_metrics": {},
    }

    # -- Behavior metrics --
    if "sycophancy" in results:
        syc = results["sycophancy"]
        summary["behavior_metrics"]["sycophancy_rate"] = syc.get("sycophancy_rate", None)
        summary["behavior_metrics"]["sycophancy_baseline_agreement"] = syc.get("baseline_agreement_rate", None)

        # Excess sycophancy: agreement on false - (1 - agreement on true)
        # This controls for the model's general tendency to agree
        baseline = syc.get("baseline_agreement_rate")
        rate = syc.get("sycophancy_rate")
        if baseline is not None and rate is not None:
            summary["behavior_metrics"]["excess_sycophancy"] = rate - (1.0 - baseline)

    if "truthfulness" in results:
        summary["behavior_metrics"]["truthfulness_score"] = results["truthfulness"].get("truthfulness_score", None)

    if "safety" in results:
        safety = results["safety"]
        summary["behavior_metrics"]["refusal_rate"] = safety.get("refusal_rate", None)
        summary["behavior_metrics"]["false_refusal_rate"] = safety.get("false_refusal_rate", None)

        # Safety score: refusal_rate - false_refusal_rate (reward correct refusal, penalize over-refusal)
        ref = safety.get("refusal_rate", 0)
        false_ref = safety.get("false_refusal_rate", 0)
        summary["behavior_metrics"]["safety_score"] = ref - false_ref

    # -- Representation metrics --
    if "probe_sycophancy" in results:
        summary["representation_metrics"]["sycophancy_probe_accuracy"] = results["probe_sycophancy"].get("accuracy")
        summary["representation_metrics"]["sycophancy_probe_auc"] = results["probe_sycophancy"].get("auc")

    if "probe_safety" in results:
        summary["representation_metrics"]["safety_probe_accuracy"] = results["probe_safety"].get("accuracy")
        summary["representation_metrics"]["safety_probe_auc"] = results["probe_safety"].get("auc")

    # Per-layer probe results
    if "probe_sycophancy_per_layer" in results:
        per_layer = results["probe_sycophancy_per_layer"]
        accs = [v["accuracy"] for v in per_layer.values()]
        if accs:
            summary["representation_metrics"]["best_sycophancy_probe_layer_accuracy"] = max(accs)
            best_layer = max(per_layer, key=lambda k: per_layer[k]["accuracy"])
            summary["representation_metrics"]["best_sycophancy_probe_layer"] = best_layer

    # Geometric analysis
    if "geometric" in results:
        geo = results["geometric"]

        if "direction_stability" in geo:
            summary["representation_metrics"]["alignment_direction_stability"] = geo["direction_stability"]

        if "layer_sensitivity" in geo:
            layer_sens = geo["layer_sensitivity"]
            cos_sims = [v.get("mean_cosine_sim", 0) for v in layer_sens.values()]
            if cos_sims:
                summary["representation_metrics"]["mean_layer_cosine_sim"] = float(np.mean(cos_sims))
                summary["representation_metrics"]["min_layer_cosine_sim"] = float(np.min(cos_sims))
                # Identify most distorted layer
                min_layer = min(layer_sens, key=lambda k: layer_sens[k].get("mean_cosine_sim", 1))
                summary["representation_metrics"]["most_distorted_layer"] = min_layer

    # -- Intervention metrics --
    if "interventions" in results:
        for behavior, behavior_data in results["interventions"].items():
            # Find alpha=0 baseline and measure max effect
            for layer_idx, layer_data in behavior_data.items():
                baseline_metric = None
                metrics = {}
                for alpha, data in layer_data.items():
                    metrics[alpha] = data["metric"]
                    if alpha == 0.0:
                        baseline_metric = data["metric"]

                if baseline_metric is not None:
                    max_reduction = baseline_metric - min(metrics.values())
                    max_increase = max(metrics.values()) - baseline_metric
                    summary["intervention_metrics"][f"{behavior}_layer{layer_idx}_max_reduction"] = max_reduction
                    summary["intervention_metrics"][f"{behavior}_layer{layer_idx}_max_increase"] = max_increase

    logger.info(f"Computed summary metrics: {len(summary['behavior_metrics'])} behavior, "
                f"{len(summary['representation_metrics'])} representation, "
                f"{len(summary['intervention_metrics'])} intervention")

    return summary


def format_metrics_table(summary: Dict[str, Any]) -> str:
    """Format summary metrics as a readable text table."""
    lines = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 60)

    for section_name, section_data in summary.items():
        if not section_data:
            continue
        lines.append("")
        lines.append(f"--- {section_name.replace('_', ' ').title()} ---")
        for key, value in section_data.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"  {key}:")
                for k2, v2 in value.items():
                    if isinstance(v2, float):
                        lines.append(f"    {k2}: {v2:.4f}")
                    else:
                        lines.append(f"    {k2}: {v2}")
            else:
                lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
