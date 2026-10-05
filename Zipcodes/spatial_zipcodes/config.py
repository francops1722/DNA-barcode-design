"""
Configuration module for spatial zipcode generation.

Defines dataclasses for chip geometry and error-correction parameters,
and provides utilities for loading configurations from JSON files.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChipGeometry:
    """Describes the physical geometry of the chip."""
    
    type: str = "rectangular"
    rows: int = 118
    cols: int = 182
    feature_size_microns: float = 14.0
    feature_spacing_microns: float = 16.0
    
    def __post_init__(self):
        """Validate chip geometry parameters."""
        if self.rows <= 0:
            raise ValueError(f"rows must be positive, got {self.rows}")
        if self.cols <= 0:
            raise ValueError(f"cols must be positive, got {self.cols}")
        if self.type not in ["rectangular", "square"]:
            logger.warning(f"Unrecognized chip type '{self.type}', assuming rectangular")
    
    @property
    def total_features(self) -> int:
        """Total number of features on the chip."""
        return self.rows * self.cols


@dataclass
class ErrorCorrection:
    """Defines error-correction constraints for spatial codes."""
    
    type: str = "edit_distance"
    distances: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    far_features_min_distance: int = 8
    
    def __post_init__(self):
        """Validate error-correction parameters."""
        if len(self.distances) != 4:
            raise ValueError(
                f"distances must contain exactly 4 values for position differences 1,2,3,4; "
                f"got {len(self.distances)}"
            )
        if any(d <= 0 for d in self.distances):
            raise ValueError(f"All distances must be positive, got {self.distances}")
        if self.far_features_min_distance <= 0:
            raise ValueError(
                f"far_features_min_distance must be positive, got {self.far_features_min_distance}"
            )
        
    def target_distance(self, position_diff: int) -> int:
        """Get target Hamming distance for a given position difference along an axis."""
        if 1 <= position_diff <= 4:
            return self.distances[position_diff - 1]
        elif position_diff >= 5:
            return self.far_features_min_distance
        else:
            return 0  # Same position


@dataclass
class BitToDNAConfig:
    """Configuration for bit-to-DNA encoding."""

    scheme: str = "2-bits-per-base"
    # None → BitToDNAEncoder picks the correct default for the scheme.
    # An explicit dict overrides the default (advanced use only).
    mapping: Optional[Dict[str, str]] = None


@dataclass
class SequenceQCConfig:
    """Quality control thresholds for DNA sequences."""
    
    gc_content_min: float = 0.3
    gc_content_max: float = 0.7
    g_content_max: float = 1.0   # No G cap by default; set e.g. 0.25 for photolithography
    max_homopolymer_length: int = 4
    reject_sequences_with_long_homopolymers: bool = True
    
    def __post_init__(self):
        """Validate QC parameters."""
        if not 0 <= self.gc_content_min <= 1:
            raise ValueError(f"gc_content_min must be in [0,1], got {self.gc_content_min}")
        if not 0 <= self.gc_content_max <= 1:
            raise ValueError(f"gc_content_max must be in [0,1], got {self.gc_content_max}")
        if self.gc_content_min > self.gc_content_max:
            raise ValueError(
                f"gc_content_min ({self.gc_content_min}) > gc_content_max ({self.gc_content_max})"
            )
        if not 0 <= self.g_content_max <= 1:
            raise ValueError(f"g_content_max must be in [0,1], got {self.g_content_max}")


@dataclass
class CodebookConfig:
    """Configuration for codebook generation."""
    
    spacer_sequence: str = "GGG"
    feature_id_format: str = "row_col"  # "row_col" or "linear"
    output_format: str = "csv"  # "csv" or "tsv"


@dataclass
class SimulationConfig:
    """Configuration for error simulation."""
    
    error_model: str = "substitution"  # "substitution" or "indel"
    per_base_error_rate: float = 0.01
    distance_metric: str = "hamming"  # "hamming" or "levenshtein"
    max_edit_distance_for_decoding: int = 3


@dataclass
class AxisCodeConfig:
    """Configuration for axis code generation algorithm."""
    
    max_attempts_per_position: int = 10000
    backtrack_depth: int = 5
    random_seed: Optional[int] = None
    tolerance: float = 0.5  # Tolerance for soft constraint violations


@dataclass
class ZipcodeConfig:
    """Complete configuration for spatial zipcode generation."""
    
    chip_geometry: ChipGeometry
    error_correction: ErrorCorrection
    bit_to_dna: BitToDNAConfig = field(default_factory=BitToDNAConfig)
    sequence_qc: SequenceQCConfig = field(default_factory=SequenceQCConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    axis_code: AxisCodeConfig = field(default_factory=AxisCodeConfig)
    
    # Derived parameters
    x_bit_length: Optional[int] = None
    y_bit_length: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived parameters."""
        # Auto-calculate bit lengths if not specified
        if self.x_bit_length is None:
            self.x_bit_length = self._estimate_bit_length(self.chip_geometry.cols)
        if self.y_bit_length is None:
            self.y_bit_length = self._estimate_bit_length(self.chip_geometry.rows)
            
        logger.info(
            f"Zipcode config initialized: "
            f"{self.chip_geometry.rows}x{self.chip_geometry.cols} chip, "
            f"X bits={self.x_bit_length}, Y bits={self.y_bit_length}"
        )
    
    def _estimate_bit_length(self, n_positions: int) -> int:
        """
        Estimate required bit length for an axis with n_positions.
        
        Uses a heuristic: need enough bits to encode all positions with
        sufficient redundancy for error correction.
        """
        # For far-features constraint, we need enough combinations
        # Rough heuristic: bit_length such that we have at least n_positions
        # combinations with minimum distance far_features_min_distance
        min_distance = self.error_correction.far_features_min_distance
        
        # Sphere-packing bound approximation
        # For binary codes of length L with minimum distance d:
        # M <= 2^L / Volume(sphere of radius (d-1)/2)
        # Volume approximation: sum over i=0 to floor((d-1)/2) of C(L,i)
        
        # Simple heuristic: start with log2(n) and add distance requirement
        import math
        base_bits = max(4, math.ceil(math.log2(n_positions)))
        
        # Add extra bits for distance constraints (empirical factor)
        extra_bits = max(0, min_distance // 2)
        
        return base_bits + extra_bits


def load_config(config_path: Path) -> ZipcodeConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        ZipcodeConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Parse nested structures
    chip_geometry = ChipGeometry(**data.get("chip_geometry", {}))
    error_correction = ErrorCorrection(**data.get("error_correction", {}))
    
    # Optional sections with defaults
    bit_to_dna = BitToDNAConfig(**data.get("bit_to_dna", {}))
    sequence_qc = SequenceQCConfig(**data.get("sequence_qc", {}))
    codebook = CodebookConfig(**data.get("codebook", {}))
    simulation = SimulationConfig(**data.get("simulation", {}))
    axis_code = AxisCodeConfig(**data.get("axis_code", {}))
    
    # Top-level optional parameters
    x_bit_length = data.get("x_bit_length")
    y_bit_length = data.get("y_bit_length")
    
    config = ZipcodeConfig(
        chip_geometry=chip_geometry,
        error_correction=error_correction,
        bit_to_dna=bit_to_dna,
        sequence_qc=sequence_qc,
        codebook=codebook,
        simulation=simulation,
        axis_code=axis_code,
        x_bit_length=x_bit_length,
        y_bit_length=y_bit_length,
    )
    
    logger.info("Configuration loaded successfully")
    return config


def save_config(config: ZipcodeConfig, config_path: Path) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: ZipcodeConfig object to save
        config_path: Path where to save the configuration
    """
    from dataclasses import asdict
    
    config_path = Path(config_path)
    
    # Convert to dict
    config_dict = asdict(config)
    
    # Write to file with pretty formatting
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
