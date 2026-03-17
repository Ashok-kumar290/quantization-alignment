"""
Linear probing for alignment features in transformer representations.

Trains logistic regression probes on hidden states to predict:
- Whether the model agrees with a false statement (sycophancy probe)
- Whether the model refuses a harmful request (safety probe)

The probe weight vector defines an "alignment direction" in activation space
that can be used for geometric analysis and causal intervention.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class AlignmentProbe:
    """
    Linear probe that identifies alignment-related directions in
    transformer hidden states.

    The weight vector of the trained logistic regression defines
    the "alignment direction" -- the linear direction in representation
    space that best separates aligned from misaligned behavior.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        regularization: float = 1.0,
        normalize: bool = True,
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.normalize = normalize

        self.scaler = StandardScaler() if normalize else None
        self.model = LogisticRegression(
            C=regularization,
            max_iter=n_epochs * 10,
            solver="lbfgs",
            random_state=42,
        )
        self._is_fitted = False

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().float().numpy()
        return np.asarray(X, dtype=np.float32)

    def _to_labels(self, y: Any) -> np.ndarray:
        """Convert labels to numpy int array."""
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy().astype(int)
        return np.asarray(y, dtype=int)

    def fit(self, X: Any, y: Any) -> "AlignmentProbe":
        """
        Train the probe.

        Args:
            X: Activations, shape [n_samples, hidden_dim].
               Accepts torch.Tensor or numpy array.
            y: Binary labels, shape [n_samples].
        """
        X_np = self._to_numpy(X)
        y_np = self._to_labels(y)

        if len(X_np) == 0:
            logger.warning("Empty training data, skipping probe training")
            return self

        if self.scaler is not None:
            X_np = self.scaler.fit_transform(X_np)

        self.model.fit(X_np, y_np)
        self._is_fitted = True

        train_acc = self.model.score(X_np, y_np)
        logger.info(f"Probe training accuracy: {train_acc:.4f}")

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary labels."""
        X_np = self._to_numpy(X)
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np)
        return self.model.predict(X_np)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        X_np = self._to_numpy(X)
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np)
        return self.model.predict_proba(X_np)

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Evaluate the probe on held-out data.

        Returns:
            Dict with accuracy, f1, and auc metrics.
        """
        X_np = self._to_numpy(X)
        y_np = self._to_labels(y)

        if self.scaler is not None:
            X_np = self.scaler.transform(X_np)

        y_pred = self.model.predict(X_np)
        y_proba = self.model.predict_proba(X_np)

        metrics = {
            "accuracy": float(accuracy_score(y_np, y_pred)),
            "f1": float(f1_score(y_np, y_pred, zero_division=0)),
        }

        # AUC requires both classes present
        if len(np.unique(y_np)) == 2:
            metrics["auc"] = float(roc_auc_score(y_np, y_proba[:, 1]))
        else:
            metrics["auc"] = 0.0

        return metrics

    def get_direction_vector(self) -> np.ndarray:
        """
        Extract the alignment direction vector from the trained probe.

        This is the weight vector of the logistic regression, which defines
        the linear direction in activation space that best separates the
        two classes (e.g., sycophantic vs. corrective).

        Returns:
            Direction vector, shape [hidden_dim].
        """
        if not self._is_fitted:
            raise RuntimeError("Probe must be fitted before extracting direction")

        direction = self.model.coef_[0].copy()

        # If we used a scaler, transform the direction back to the original space
        if self.scaler is not None:
            # w_original = w_scaled / scale
            direction = direction / self.scaler.scale_

        # Normalize to unit vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction


class LayerwiseProbeAnalysis:
    """
    Trains probes at every transformer layer to identify where
    alignment information is encoded.
    """

    def __init__(
        self,
        regularization: float = 1.0,
        normalize: bool = True,
        test_fraction: float = 0.2,
    ):
        self.regularization = regularization
        self.normalize = normalize
        self.test_fraction = test_fraction
        self.probes: Dict[str, AlignmentProbe] = {}
        self.results: Dict[str, Dict] = {}

    def train_all_layers(
        self,
        per_layer_activations: Dict[str, Any],
        labels: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train a separate probe at each layer.

        Args:
            per_layer_activations: Dict mapping layer name -> activations tensor
                with shape [n_samples, hidden_dim].
            labels: Binary labels, shape [n_samples].

        Returns:
            Dict mapping layer name -> evaluation metrics.
        """
        if isinstance(labels, list):
            labels = np.array(labels, dtype=int)
        elif isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy().astype(int)

        n = len(labels)
        split = int(n * (1 - self.test_fraction))
        indices = np.random.RandomState(42).permutation(n)
        train_idx, test_idx = indices[:split], indices[split:]

        results = {}

        for layer_name, activations in per_layer_activations.items():
            if isinstance(activations, torch.Tensor):
                activations = activations.cpu().float().numpy()

            X_train = activations[train_idx]
            y_train = labels[train_idx]
            X_test = activations[test_idx]
            y_test = labels[test_idx]

            probe = AlignmentProbe(
                input_dim=activations.shape[1],
                regularization=self.regularization,
                normalize=self.normalize,
            )
            probe.fit(X_train, y_train)
            metrics = probe.evaluate(X_test, y_test)

            self.probes[layer_name] = probe
            results[layer_name] = metrics

            logger.info(f"Layer {layer_name}: accuracy={metrics['accuracy']:.3f}, auc={metrics['auc']:.3f}")

        self.results = results
        return results

    def get_best_layer(self) -> Tuple[str, Dict]:
        """Return the layer with highest probe accuracy."""
        if not self.results:
            raise RuntimeError("Must train probes first")
        best_layer = max(self.results, key=lambda k: self.results[k]["accuracy"])
        return best_layer, self.results[best_layer]

    def get_direction_vectors(self) -> Dict[str, np.ndarray]:
        """Get alignment direction vectors from all layers."""
        return {
            name: probe.get_direction_vector()
            for name, probe in self.probes.items()
            if probe._is_fitted
        }

    def get_layer_sensitivity_curve(self) -> Tuple[List[str], List[float]]:
        """Return ordered (layer_names, accuracies) for plotting."""
        ordered = sorted(self.results.items(), key=lambda x: _extract_layer_num(x[0]))
        names = [item[0] for item in ordered]
        accs = [item[1]["accuracy"] for item in ordered]
        return names, accs


def _extract_layer_num(layer_name: str) -> int:
    """Extract numeric layer index from a layer name like 'model.layers.15'."""
    parts = layer_name.split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0
