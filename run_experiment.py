#!/usr/bin/env python3
"""
Main experiment pipeline for quantization & alignment research.

This script orchestrates the full research pipeline:
1. Load model with specified precision
2. Run evaluation prompts
3. Collect activations (per-layer)
4. Train probes (per-layer + single best)
5. Run geometric analysis
6. Run causal interventions
7. Compute metrics
8. Generate plots

Usage:
    python run_experiment.py --model mistral --precision 4bit
    python run_experiment.py --model llama3 --precision 8bit --skip-probing
    python run_experiment.py --model qwen --precision 4bit --intervention-only
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np

# Project imports
from models.model_loader import ModelLoader, QuantizationConfig
from models.activation_collector import ActivationCollector
from datasets.sycophancy_dataset import SycophancyDataset
from datasets.truthfulness_dataset import TruthfulnessDataset
from datasets.safety_dataset import SafetyDataset
from experiments.evaluator import AlignmentEvaluator
from probes.linear_probe import AlignmentProbe, LayerwiseProbeAnalysis
from analysis.geometric import GeometricAnalyzer
from interventions.steering import ActivationSteering
from utils.visualization import ResultsVisualizer
from utils.metrics import compute_metrics, format_metrics_table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for experiment runs."""

    # Model configurations
    MODEL_CONFIGS = {
        'llama3': {
            'name': 'meta-llama/Llama-3-8B',
            'default_precision': 'fp16',
            'supported_precisions': ['fp16', '8bit', '4bit', '3bit']
        },
        'mistral': {
            'name': 'mistralai/Mistral-7B-v0.1',
            'default_precision': 'fp16',
            'supported_precisions': ['fp16', '8bit', '4bit', '3bit']
        },
        'qwen': {
            'name': 'Qwen/Qwen2-7B',
            'default_precision': 'fp16',
            'supported_precisions': ['fp16', '8bit', '4bit', '3bit']
        }
    }

    # Precision settings
    PRECISION_CONFIGS = {
        'fp16': {'load_in_8bit': False, 'load_in_4bit': False, 'torch_dtype': torch.float16},
        '8bit': {'load_in_8bit': True, 'load_in_4bit': False, 'torch_dtype': torch.float16},
        '4bit': {'load_in_8bit': False, 'load_in_4bit': True, 'torch_dtype': torch.float16},
        '3bit': {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4'}
    }

    # Dataset sizes for evaluation
    SYCOPHANCY_N_SAMPLES = 200
    TRUTHFULNESS_N_SAMPLES = 100
    SAFETY_N_SAMPLES = 100

    # Probe training
    PROBE_TRAIN_SPLIT = 0.8
    PROBE_LR = 1e-3
    PROBE_EPOCHS = 50

    # Intervention settings
    INTERVENTION_LAYERS = [15, 20, 25, 30]  # Late layers typically most important
    INTERVENTION_ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # Output settings
    RESULTS_DIR = Path('results')
    FIGURE_DIR = RESULTS_DIR / 'figures'
    DATA_DIR = RESULTS_DIR / 'data'

    def __init__(self, model: str, precision: str, **kwargs):
        self.model_name = model
        self.precision = precision
        self.experiment_name = f"{model}_{precision}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate configuration
        self._validate()

        # Create output directories
        self._setup_directories()

    def _validate(self):
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[self.model_name]
        if self.precision not in config['supported_precisions']:
            raise ValueError(
                f"Precision {self.precision} not supported for {self.model_name}. "
                f"Supported: {config['supported_precisions']}"
            )

    def _setup_directories(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def get_model_path(self) -> str:
        return self.MODEL_CONFIGS[self.model_name]['name']

    def get_quantization_config(self) -> Dict[str, Any]:
        base_config = self.PRECISION_CONFIGS[self.precision].copy()
        if self.precision in ('4bit', '3bit'):
            base_config.update({
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': True,
            })
        return base_config


class ExperimentPipeline:
    """Main experiment pipeline orchestrator."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.activations = {}          # last-layer activations (legacy)
        self.per_layer_activations = {} # per-layer activations for probing
        self.probes = {}
        self.layerwise_probes = {}     # LayerwiseProbeAnalysis objects
        self.results = {}

    def run(self,
            skip_evaluation: bool = False,
            skip_probing: bool = False,
            skip_intervention: bool = False,
            intervention_only: bool = False):
        """Run the full experiment pipeline."""
        logger.info(f"Starting experiment: {self.config.experiment_name}")

        try:
            # Step 1: Load model
            if not intervention_only:
                self._load_model()

            # Step 2: Behavioral evaluation
            if not skip_evaluation and not intervention_only:
                self._run_evaluation()

            # Step 3: Collect activations (per-layer)
            if not skip_probing and not intervention_only:
                self._collect_activations()

            # Step 4: Train probes (per-layer + best)
            if not skip_probing and not intervention_only:
                self._train_probes()

            # Step 5: Geometric analysis
            if not skip_probing and not intervention_only:
                self._run_geometric_analysis()

            # Step 6: Causal interventions
            if not skip_intervention:
                self._run_interventions()

            # Step 7: Compute aggregate metrics
            self._compute_metrics()

            # Step 8: Generate visualizations
            self._generate_visualizations()

            # Step 9: Save results
            self._save_results()

            logger.info("Experiment completed successfully!")
            return self.results

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

    def _load_model(self):
        logger.info(f"Loading model: {self.config.get_model_path()} with {self.config.precision}")

        loader = ModelLoader(
            model_name=self.config.get_model_path(),
            quantization_config=self.config.get_quantization_config()
        )

        self.model, self.tokenizer = loader.load()
        logger.info("Model loaded successfully")

    def _run_evaluation(self):
        logger.info("Running alignment evaluation...")

        evaluator = AlignmentEvaluator(
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Sycophancy evaluation
        sycophancy_ds = SycophancyDataset(n_samples=self.config.SYCOPHANCY_N_SAMPLES)
        sycophancy_results = evaluator.evaluate_sycophancy(sycophancy_ds)
        self.results['sycophancy'] = sycophancy_results

        # Truthfulness evaluation
        truthfulness_ds = TruthfulnessDataset(n_samples=self.config.TRUTHFULNESS_N_SAMPLES)
        truthfulness_results = evaluator.evaluate_truthfulness(truthfulness_ds)
        self.results['truthfulness'] = truthfulness_results

        # Safety evaluation
        safety_ds = SafetyDataset(n_samples=self.config.SAFETY_N_SAMPLES)
        safety_results = evaluator.evaluate_safety(safety_ds)
        self.results['safety'] = safety_results

        logger.info(f"Sycophancy rate: {sycophancy_results['sycophancy_rate']:.3f}")
        logger.info(f"Truthfulness score: {truthfulness_results['truthfulness_score']:.3f}")
        logger.info(f"Refusal rate: {safety_results['refusal_rate']:.3f}")

    def _collect_activations(self):
        """Collect per-layer and last-layer activations for probing."""
        logger.info("Collecting activations...")

        collector = ActivationCollector(self.model, self.tokenizer)

        # Sycophancy: per-layer activations for layerwise probe training
        sycophancy_ds = SycophancyDataset(n_samples=self.config.SYCOPHANCY_N_SAMPLES)
        syc_all_layers = collector.collect_all_layers(sycophancy_ds)
        self.per_layer_activations['sycophancy'] = syc_all_layers

        # Also store last-layer for backward-compatible single-probe training
        if syc_all_layers['per_layer']:
            last_layer_name = sorted(syc_all_layers['per_layer'].keys())[-1]
            self.activations['sycophancy'] = {
                'activations': syc_all_layers['per_layer'][last_layer_name],
                'labels': syc_all_layers['labels'],
                'hidden_size': syc_all_layers['hidden_size'],
                'n_layers': syc_all_layers['n_layers'],
            }

        # Safety: per-layer activations
        safety_ds = SafetyDataset(n_samples=self.config.SAFETY_N_SAMPLES)
        safety_all_layers = collector.collect_all_layers(safety_ds)
        self.per_layer_activations['safety'] = safety_all_layers

        if safety_all_layers['per_layer']:
            last_layer_name = sorted(safety_all_layers['per_layer'].keys())[-1]
            self.activations['safety'] = {
                'activations': safety_all_layers['per_layer'][last_layer_name],
                'labels': safety_all_layers['labels'],
                'hidden_size': safety_all_layers['hidden_size'],
                'n_layers': safety_all_layers['n_layers'],
            }

        n_layers = len(syc_all_layers.get('per_layer', {}))
        logger.info(f"Collected per-layer activations from {n_layers} layers")

    def _train_probes(self):
        """Train per-layer probes and single best-layer probes."""
        logger.info("Training alignment probes...")

        for behavior in ('sycophancy', 'safety'):
            if behavior not in self.per_layer_activations:
                continue

            data = self.per_layer_activations[behavior]
            per_layer = data.get('per_layer', {})
            labels = data.get('labels')

            if not per_layer or labels is None:
                logger.warning(f"No per-layer data for {behavior}, skipping")
                continue

            # Train per-layer probes
            layerwise = LayerwiseProbeAnalysis(test_fraction=1 - self.config.PROBE_TRAIN_SPLIT)
            per_layer_results = layerwise.train_all_layers(per_layer, labels)

            self.layerwise_probes[behavior] = layerwise
            self.results[f'probe_{behavior}_per_layer'] = per_layer_results

            # Get best layer probe as the primary probe for interventions
            best_layer, best_metrics = layerwise.get_best_layer()
            self.probes[behavior] = layerwise.probes[best_layer]
            self.results[f'probe_{behavior}'] = best_metrics

            logger.info(f"{behavior} probe: best layer={best_layer}, "
                        f"accuracy={best_metrics['accuracy']:.3f}")

    def _run_geometric_analysis(self):
        """Run geometric analysis on collected activations."""
        logger.info("Running geometric analysis...")
        geo_results = {}

        # Analyze activation norms per layer
        for behavior in ('sycophancy', 'safety'):
            data = self.per_layer_activations.get(behavior, {})
            per_layer = data.get('per_layer', {})
            if per_layer:
                norms = GeometricAnalyzer.compute_activation_norms(
                    {k: v.cpu().numpy() for k, v in per_layer.items()}
                )
                geo_results[f'{behavior}_activation_norms'] = norms

        # PCA on sycophancy activations (best layer)
        if 'sycophancy' in self.layerwise_probes:
            lw = self.layerwise_probes['sycophancy']
            best_layer, _ = lw.get_best_layer()
            data = self.per_layer_activations['sycophancy']
            acts = data['per_layer'].get(best_layer)
            labels = data.get('labels')

            if acts is not None:
                pca_result = GeometricAnalyzer.compute_pca(
                    acts.cpu().numpy(),
                    n_components=2,
                    labels=np.array(labels) if labels else None,
                )
                geo_results['pca_sycophancy'] = pca_result

        self.results['geometric'] = geo_results
        logger.info("Geometric analysis completed")

    def _run_interventions(self):
        logger.info("Running intervention experiments...")

        alignment_directions = {}

        if 'sycophancy' in self.probes:
            alignment_directions['sycophancy'] = self.probes['sycophancy'].get_direction_vector()

        if 'safety' in self.probes:
            alignment_directions['safety'] = self.probes['safety'].get_direction_vector()

        if not alignment_directions:
            logger.warning("No alignment directions available for intervention")
            return

        # Clamp intervention layers to model's actual layer count
        n_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        valid_layers = [l for l in self.config.INTERVENTION_LAYERS if l < n_layers]
        if not valid_layers:
            valid_layers = [n_layers // 2, int(n_layers * 0.75), n_layers - 2]

        steering = ActivationSteering(
            model=self.model,
            tokenizer=self.tokenizer,
            alignment_directions=alignment_directions
        )

        intervention_results = steering.run_intervention_sweep(
            layers=valid_layers,
            alphas=self.config.INTERVENTION_ALPHAS
        )

        self.results['interventions'] = intervention_results
        logger.info("Intervention experiments completed")

    def _compute_metrics(self):
        logger.info("Computing metrics...")
        metrics = compute_metrics(self.results)
        self.results['summary'] = metrics
        print(format_metrics_table(metrics))

    def _generate_visualizations(self):
        logger.info("Generating visualizations...")

        visualizer = ResultsVisualizer(
            results=self.results,
            output_dir=self.config.FIGURE_DIR
        )

        visualizer.generate_all_plots()

        # PCA projection if available
        geo = self.results.get('geometric', {})
        if 'pca_sycophancy' in geo:
            visualizer.plot_pca_projections(
                geo['pca_sycophancy'],
                title="PCA of Sycophancy Activations"
            )

        logger.info(f"Visualizations saved to {self.config.FIGURE_DIR}")

    def _save_results(self):
        logger.info("Saving results...")

        results_file = self.config.DATA_DIR / f"{self.config.experiment_name}_results.json"

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        # Filter out large activation tensors from saved results
        saveable = {k: v for k, v in self.results.items()
                    if k not in ('activations_per_layer',)}
        serializable_results = convert_to_serializable(saveable)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run quantization & alignment experiment'
    )

    parser.add_argument(
        '--model', type=str, default='mistral',
        choices=['llama3', 'mistral', 'qwen'],
        help='Model to evaluate'
    )

    parser.add_argument(
        '--precision', type=str, default='4bit',
        choices=['fp16', '8bit', '4bit', '3bit'],
        help='Model precision/quantization level'
    )

    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip behavioral evaluation')
    parser.add_argument('--skip-probing', action='store_true',
                        help='Skip probe training')
    parser.add_argument('--skip-intervention', action='store_true',
                        help='Skip intervention experiments')
    parser.add_argument('--intervention-only', action='store_true',
                        help='Only run interventions (requires pre-computed probes)')

    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        model=args.model,
        precision=args.precision,
        results_dir=args.output_dir
    )

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(0)

    pipeline = ExperimentPipeline(config)
    results = pipeline.run(
        skip_evaluation=args.skip_evaluation,
        skip_probing=args.skip_probing,
        skip_intervention=args.skip_intervention,
        intervention_only=args.intervention_only
    )

    print(f"\n{'='*60}")
    print(f"Experiment completed: {config.experiment_name}")
    print(f"Results saved to: {config.DATA_DIR}")
    print(f"Figures saved to: {config.FIGURE_DIR}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
