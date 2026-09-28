"""Object detection and segmentation with optional inference backends."""

from .detection import Detector
from .utils import DetectionResults, DetectedObject

__all__ = ["Detector", "DetectionResults", "DetectedObject"]
