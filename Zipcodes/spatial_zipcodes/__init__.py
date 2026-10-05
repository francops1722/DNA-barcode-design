"""
Spatial Zipcodes - A package for generating error-correcting spatial barcodes
for photolithography-based spatial transcriptomics platforms.

This package implements 2D Gray-code-like spatial barcoding with configurable
error-correction parameters based on edit distance constraints.
"""

__version__ = "0.1.0"
__author__ = "Spatial Transcriptomics Tools"

from .config import ChipGeometry, ErrorCorrection, ZipcodeConfig, load_config
from .codebook import build_codebook, export_codebook

__all__ = [
    "ChipGeometry",
    "ErrorCorrection", 
    "ZipcodeConfig",
    "load_config",
    "build_codebook",
    "export_codebook",
]
