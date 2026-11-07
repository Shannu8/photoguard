"""Forward-compatible service interfaces for future research phases (Phases 3 to 8).

This module defines standard protocols for future extensions:
- Phase 3: Spatial Transformations (rotation, affine shearing, scaling)
- Phase 4: Mathematical Perturbations (adversarial perturbations, frequency watermarks)
- Phase 5: Robustness Testing (JPEG re-compression, gaussian noise, spatial downsampling)
- Phase 6 & 7: AI Protection & Deepfake Detection Models
- Phase 8: Benchmark and Evaluation Harness

Existing Phase 1 and 2 services interact with images strictly as standard NumPy ndarrays,
ensuring complete separation of concerns and zero rewrite requirements for future phases.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ImageTransformerInterface(ABC):
    """Interface for Phase 3: Image spatial transformations."""

    @abstractmethod
    def transform(self, image_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply geometric transformation to image array (H, W, C)."""
        pass


class ImageProtectionInterface(ABC):
    """Interface for Phase 4 & 6: Mathematical perturbation and AI protection."""

    @abstractmethod
    def generate_perturbation(
        self, image_array: np.ndarray, config: Dict[str, Any]
    ) -> np.ndarray:
        """Generate subtle mathematical perturbation matrix delta of shape (H, W, C)."""
        pass

    @abstractmethod
    def apply_protection(
        self, image_array: np.ndarray, config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply protection and return (protected_array, metadata)."""
        pass


class RobustnessEvaluatorInterface(ABC):
    """Interface for Phase 5: Compression and noise robustness evaluation."""

    @abstractmethod
    def evaluate_robustness(
        self, original: np.ndarray, perturbed: np.ndarray, attack_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Test degradation metrics under compression, resizing, or filtering."""
        pass


class DeepfakeDetectorInterface(ABC):
    """Interface for Phase 7: Deepfake detection and spectral fingerprinting."""

    @abstractmethod
    def analyze_authenticity(
        self, image_array: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute authenticity score and spectral anomaly metrics."""
        pass

