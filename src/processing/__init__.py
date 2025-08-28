"""
Processing module for rPPG signal processing and face detection
"""

from .face_detection import FaceDetectionProcessor
from .signal_quality import SignalQualityAssessment

__all__ = [
    'FaceDetectionProcessor',
    'SignalQualityAssessment'
]
