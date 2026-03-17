"""
Geometric analysis of internal representations.

Provides tools for analyzing how quantization distorts the geometry
of alignment-related representations:
- Cosine similarity between alignment direction vectors
- PCA of hidden states
- Centered Kernel Alignment (CKA) for representational similarity
- Layer-wise sensitivity to quantization
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GeometricAnalyzer:
    """
    Analyzes the geometry of internal representations to detect
    distortion introduced by quantization.
    """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a, b: 1-D arrays.

        Returns:
            Cosine similarity in [-1, 1].
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def cosine_similarity_matrix(vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise cosine similarity matrix between named vectors.

        Args:
            vectors: Dict mapping label -> 1-D direction vector.

        Returns:
            (similarity_matrix, labels) where similarity_matrix[i,j] is
            the cosine similarity between vectors[labels[i]] and vectors[labels[j]].
        """
        labels = sorted(vectors.keys())
        n = len(labels)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = GeometricAnalyzer.cosine_similarity(
                    vectors[labels[i]], vectors[labels[j]]
                )

        return sim_matrix, labels

    @staticmethod
    def compute_pca(
        activations: np.ndarray,
        n_components: int = 2,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        PCA projection of hidden states.

        Args:
            activations: [n_samples, hidden_dim] array.
            n_components: Number of PCA components.
            labels: Optional labels for coloring.

        Returns:
            Dict with 'projections', 'explained_variance_ratio',
            'components', and 'labels'.
        """
        activations = np.asarray(activations, dtype=np.float64)

        # Center the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(activations)

        pca = PCA(n_components=n_components, random_state=42)
        projections = pca.fit_transform(X_scaled)

        return {
            "projections": projections,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_,
            "labels": labels,
            "n_components": n_components,
        }

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Linear Centered Kernel Alignment (CKA) between two
        representation matrices.

        CKA measures representational similarity independently of
        rotation and isotropic scaling, making it suitable for comparing
        activations across different models or quantization levels.

        Reference: Kornblith et al., "Similarity of Neural Network
        Representations Revisited" (ICML 2019).

        Args:
            X: [n_samples, d1] activation matrix.
            Y: [n_samples, d2] activation matrix.

        Returns:
            CKA similarity in [0, 1].
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # Center
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Compute HSIC with linear kernel
        # HSIC(K, L) = (1/n^2) * tr(KHLH) where H = I - (1/n)*11^T
        # For linear kernel this simplifies to: ||Y^T X||_F^2
        YtX = Y.T @ X
        hsic_xy = np.sum(YtX ** 2)

        XtX = X.T @ X
        hsic_xx = np.sum(XtX ** 2)

        YtY = Y.T @ Y
        hsic_yy = np.sum(YtY ** 2)

        denom = np.sqrt(hsic_xx * hsic_yy)
        if denom == 0:
            return 0.0

        return float(hsic_xy / denom)

    @staticmethod
    def representational_similarity_matrix(
        activations_by_condition: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise CKA similarity between activation sets.

        Args:
            activations_by_condition: Dict mapping condition label
                (e.g., 'fp16', '4bit') -> [n_samples, hidden_dim] array.

        Returns:
            (cka_matrix, labels).
        """
        labels = sorted(activations_by_condition.keys())
        n = len(labels)
        cka_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                cka_matrix[i, j] = GeometricAnalyzer.linear_cka(
                    activations_by_condition[labels[i]],
                    activations_by_condition[labels[j]],
                )

        return cka_matrix, labels

    @staticmethod
    def alignment_vector_stability(
        directions_by_precision: Dict[str, np.ndarray],
        reference_precision: str = "fp16",
    ) -> Dict[str, float]:
        """
        Measure how stable alignment direction vectors are across
        quantization levels relative to a baseline.

        Args:
            directions_by_precision: Dict mapping precision -> direction vector.
            reference_precision: Which precision to use as baseline.

        Returns:
            Dict mapping precision -> cosine similarity with reference.
        """
        if reference_precision not in directions_by_precision:
            logger.warning(f"Reference precision {reference_precision} not found")
            return {}

        ref = directions_by_precision[reference_precision]
        stability = {}

        for precision, direction in directions_by_precision.items():
            stability[precision] = GeometricAnalyzer.cosine_similarity(ref, direction)

        return stability

    @staticmethod
    def layer_sensitivity(
        activations_baseline: Dict[str, np.ndarray],
        activations_quantized: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-layer representation distance between baseline
        and quantized model activations.

        Metrics per layer:
        - mean_cosine_sim: average cosine similarity across samples
        - mean_l2_distance: average L2 distance across samples
        - cka: CKA similarity for the layer

        Args:
            activations_baseline: Dict of layer_name -> [n, d] baseline activations.
            activations_quantized: Dict of layer_name -> [n, d] quantized activations.

        Returns:
            Dict of layer_name -> metric dict.
        """
        common_layers = set(activations_baseline.keys()) & set(activations_quantized.keys())
        results = {}

        for layer in sorted(common_layers):
            X_base = np.asarray(activations_baseline[layer], dtype=np.float64)
            X_quant = np.asarray(activations_quantized[layer], dtype=np.float64)

            n = min(len(X_base), len(X_quant))
            X_base = X_base[:n]
            X_quant = X_quant[:n]

            # Per-sample cosine similarity
            cos_sims = []
            l2_dists = []
            for i in range(n):
                cos_sims.append(GeometricAnalyzer.cosine_similarity(X_base[i], X_quant[i]))
                l2_dists.append(float(np.linalg.norm(X_base[i] - X_quant[i])))

            results[layer] = {
                "mean_cosine_sim": float(np.mean(cos_sims)),
                "std_cosine_sim": float(np.std(cos_sims)),
                "mean_l2_distance": float(np.mean(l2_dists)),
                "std_l2_distance": float(np.std(l2_dists)),
                "cka": GeometricAnalyzer.linear_cka(X_base, X_quant),
            }

        return results

    @staticmethod
    def compute_activation_norms(
        activations: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-layer activation norm statistics.

        Useful for detecting whether quantization causes activation
        magnitude changes that could affect downstream behavior.
        """
        results = {}
        for layer, X in activations.items():
            X = np.asarray(X, dtype=np.float64)
            norms = np.linalg.norm(X, axis=1)
            results[layer] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
            }
        return results
