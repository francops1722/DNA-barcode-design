"""
Simulation module for error modeling and decoding validation.

Simulates sequencing errors and tests decoding accuracy.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

from .codebook import Feature, Codebook
from .distance_analysis import levenshtein_distance

logger = logging.getLogger(__name__)


class ErrorModel:
    """Base class for error models."""
    
    def apply(self, sequence: str) -> str:
        """Apply errors to a sequence."""
        raise NotImplementedError


class SubstitutionErrorModel(ErrorModel):
    """
    Simple per-base substitution error model.
    
    Each base has a probability of being substituted with another random base.
    """
    
    def __init__(self, error_rate: float, random_seed: Optional[int] = None):
        """
        Initialize substitution error model.
        
        Args:
            error_rate: Per-base error probability
            random_seed: Random seed
        """
        self.error_rate = error_rate
        self.rng = random.Random(random_seed)
        self.bases = ['A', 'C', 'G', 'T']
    
    def apply(self, sequence: str) -> str:
        """Apply substitution errors to sequence."""
        sequence = sequence.upper()
        result = []
        
        for base in sequence:
            if self.rng.random() < self.error_rate:
                # Substitute with a different base
                alternatives = [b for b in self.bases if b != base]
                result.append(self.rng.choice(alternatives))
            else:
                result.append(base)
        
        return ''.join(result)


class IndelErrorModel(ErrorModel):
    """
    Insertion/deletion error model.
    
    Each position has a probability of insertion or deletion.
    """
    
    def __init__(
        self,
        substitution_rate: float = 0.01,
        insertion_rate: float = 0.005,
        deletion_rate: float = 0.005,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize indel error model.
        
        Args:
            substitution_rate: Per-base substitution probability
            insertion_rate: Per-base insertion probability
            deletion_rate: Per-base deletion probability
            random_seed: Random seed
        """
        self.substitution_rate = substitution_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.rng = random.Random(random_seed)
        self.bases = ['A', 'C', 'G', 'T']
    
    def apply(self, sequence: str) -> str:
        """Apply substitution, insertion, and deletion errors."""
        sequence = sequence.upper()
        result = []
        
        for base in sequence:
            # Check for deletion
            if self.rng.random() < self.deletion_rate:
                continue  # Skip this base (deletion)
            
            # Check for substitution
            if self.rng.random() < self.substitution_rate:
                alternatives = [b for b in self.bases if b != base]
                result.append(self.rng.choice(alternatives))
            else:
                result.append(base)
            
            # Check for insertion after this base
            if self.rng.random() < self.insertion_rate:
                result.append(self.rng.choice(self.bases))
        
        return ''.join(result)


class ZipcodeDecoder:
    """
    Decodes error-corrupted zip codes back to (row, col) positions.
    
    Uses nearest-neighbor decoding with configurable distance metric
    and maximum allowed distance.
    """
    
    def __init__(
        self,
        codebook: Codebook,
        distance_metric: str = "hamming",
        max_edit_distance: int = 3,
        spacer: str = "GGG",
    ):
        """
        Initialize decoder.
        
        Args:
            codebook: Reference codebook
            distance_metric: "hamming" or "levenshtein"
            max_edit_distance: Maximum distance to accept a match
            spacer: Spacer sequence (for splitting X and Y)
        """
        self.codebook = codebook
        self.distance_metric = distance_metric
        self.max_edit_distance = max_edit_distance
        self.spacer = spacer
        
        # Pre-compute reference sequences
        self.x_sequences = codebook.x_sequences
        self.y_sequences = codebook.y_sequences
        
        logger.info(
            f"Initialized decoder: {distance_metric} distance, "
            f"max_distance={max_edit_distance}"
        )
    
    def _compute_distance(self, seq1: str, seq2: str) -> int:
        """Compute distance between two sequences."""
        if self.distance_metric == "hamming":
            if len(seq1) != len(seq2):
                return float('inf')  # Can't compute Hamming for different lengths
            return sum(b1 != b2 for b1, b2 in zip(seq1, seq2))
        elif self.distance_metric == "levenshtein":
            return levenshtein_distance(seq1, seq2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _decode_component(
        self,
        observed: str,
        reference_sequences: List[str],
    ) -> Tuple[Optional[int], int, bool]:
        """
        Decode a single component (X or Y) by finding nearest reference.
        
        Args:
            observed: Observed (possibly corrupted) sequence
            reference_sequences: List of reference sequences
            
        Returns:
            Tuple of (best_index, best_distance, is_ambiguous)
        """
        best_distance = float('inf')
        best_indices = []
        
        for i, ref_seq in enumerate(reference_sequences):
            dist = self._compute_distance(observed, ref_seq)
            
            if dist < best_distance:
                best_distance = dist
                best_indices = [i]
            elif dist == best_distance:
                best_indices.append(i)
        
        # Check if distance is acceptable
        if best_distance > self.max_edit_distance:
            return None, best_distance, False
        
        # Check for ambiguity (ties)
        is_ambiguous = len(best_indices) > 1
        
        best_index = best_indices[0] if best_indices else None
        
        return best_index, best_distance, is_ambiguous
    
    def decode(self, observed_zipcode: str) -> Dict[str, any]:
        """
        Decode an observed zip code sequence.
        
        Args:
            observed_zipcode: Observed (possibly corrupted) zip code
            
        Returns:
            Dictionary with decoding results:
                - 'success': bool
                - 'row': int or None
                - 'col': int or None
                - 'x_distance': int
                - 'y_distance': int
                - 'ambiguous': bool
                - 'reason': str (if failed)
        """
        result = {
            'success': False,
            'row': None,
            'col': None,
            'x_distance': None,
            'y_distance': None,
            'ambiguous': False,
            'reason': '',
        }
        
        # Try to split by spacer
        parts = observed_zipcode.split(self.spacer)
        
        if len(parts) == 2:
            x_obs, y_obs = parts
        else:
            # Spacer might be corrupted, try approximate splitting
            # Use expected lengths
            x_len = len(self.x_sequences[0])
            y_len = len(self.y_sequences[0])
            spacer_len = len(self.spacer)
            
            if len(observed_zipcode) >= x_len + y_len:
                # Try to extract X and Y based on expected positions
                x_obs = observed_zipcode[:x_len]
                y_obs = observed_zipcode[-(y_len):]
            else:
                result['reason'] = "Cannot parse zip code (too short or spacer corrupted)"
                return result
        
        # Decode X component
        x_idx, x_dist, x_ambig = self._decode_component(x_obs, self.x_sequences)
        result['x_distance'] = x_dist
        
        # Decode Y component
        y_idx, y_dist, y_ambig = self._decode_component(y_obs, self.y_sequences)
        result['y_distance'] = y_dist
        
        # Check if decoding succeeded
        if x_idx is None or y_idx is None:
            result['reason'] = "Distance to nearest code exceeds maximum"
            return result
        
        if x_ambig or y_ambig:
            result['ambiguous'] = True
            result['reason'] = "Ambiguous (tie in nearest-neighbor)"
        
        result['success'] = True
        result['col'] = x_idx
        result['row'] = y_idx
        
        return result


class ZipcodeSimulator:
    """
    Simulates sequencing errors and evaluates decoding performance.
    """
    
    def __init__(
        self,
        codebook: Codebook,
        error_model: ErrorModel,
        decoder: ZipcodeDecoder,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize simulator.
        
        Args:
            codebook: Reference codebook
            error_model: Error model to apply
            decoder: Decoder for corrupted sequences
            random_seed: Random seed for sampling
        """
        self.codebook = codebook
        self.error_model = error_model
        self.decoder = decoder
        self.rng = random.Random(random_seed)
        
        logger.info("Initialized zipcode simulator")
    
    def simulate_single(self, feature: Feature) -> Dict[str, any]:
        """
        Simulate errors and decoding for a single feature.
        
        Args:
            feature: Feature to simulate
            
        Returns:
            Dictionary with simulation results
        """
        # Apply errors
        corrupted_zipcode = self.error_model.apply(feature.zip_seq)
        
        # Decode
        decode_result = self.decoder.decode(corrupted_zipcode)
        
        # Evaluate
        result = {
            'true_row': feature.row,
            'true_col': feature.col,
            'decoded_row': decode_result['row'],
            'decoded_col': decode_result['col'],
            'success': decode_result['success'],
            'ambiguous': decode_result['ambiguous'],
            'x_distance': decode_result['x_distance'],
            'y_distance': decode_result['y_distance'],
        }
        
        # Classify result
        if not decode_result['success']:
            result['outcome'] = 'failed'
        elif decode_result['ambiguous']:
            result['outcome'] = 'ambiguous'
        elif decode_result['row'] == feature.row and decode_result['col'] == feature.col:
            result['outcome'] = 'correct'
        else:
            # Misassigned - check if near or far
            row_diff = abs(decode_result['row'] - feature.row)
            col_diff = abs(decode_result['col'] - feature.col)
            spatial_distance = max(row_diff, col_diff)
            
            result['spatial_distance'] = spatial_distance
            
            if spatial_distance <= 2:
                result['outcome'] = 'near_error'
            else:
                result['outcome'] = 'far_error'
        
        return result
    
    def simulate_batch(
        self,
        n_samples: int = 1000,
        sample_all: bool = False,
    ) -> Dict[str, any]:
        """
        Simulate errors and decoding for multiple features.
        
        Args:
            n_samples: Number of features to sample
            sample_all: If True, simulate all features (ignore n_samples)
            
        Returns:
            Dictionary with aggregate results
        """
        if sample_all:
            features_to_test = self.codebook.features
        else:
            features_to_test = self.rng.sample(
                self.codebook.features,
                min(n_samples, len(self.codebook.features))
            )
        
        logger.info(f"Simulating {len(features_to_test)} features...")
        
        outcomes = defaultdict(int)
        individual_results = []
        
        for i, feature in enumerate(features_to_test):
            if (i + 1) % 100 == 0:
                logger.info(f"  Simulated {i + 1}/{len(features_to_test)}")
            
            result = self.simulate_single(feature)
            individual_results.append(result)
            outcomes[result['outcome']] += 1
        
        # Aggregate statistics
        n_total = len(features_to_test)
        
        aggregate = {
            'n_simulated': n_total,
            'n_correct': outcomes['correct'],
            'n_near_error': outcomes['near_error'],
            'n_far_error': outcomes['far_error'],
            'n_ambiguous': outcomes['ambiguous'],
            'n_failed': outcomes['failed'],
            'accuracy': outcomes['correct'] / n_total if n_total > 0 else 0,
            'near_error_rate': outcomes['near_error'] / n_total if n_total > 0 else 0,
            'far_error_rate': outcomes['far_error'] / n_total if n_total > 0 else 0,
            'ambiguous_rate': outcomes['ambiguous'] / n_total if n_total > 0 else 0,
            'failed_rate': outcomes['failed'] / n_total if n_total > 0 else 0,
            'individual_results': individual_results,
        }
        
        # Distance statistics for successful decodings
        successful = [r for r in individual_results if r['success']]
        if successful:
            x_distances = [r['x_distance'] for r in successful]
            y_distances = [r['y_distance'] for r in successful]
            
            aggregate['x_distance_mean'] = np.mean(x_distances)
            aggregate['y_distance_mean'] = np.mean(y_distances)
            aggregate['x_distance_max'] = np.max(x_distances)
            aggregate['y_distance_max'] = np.max(y_distances)
        
        return aggregate
    
    def print_summary(self, results: Dict[str, any]):
        """Print human-readable summary of simulation results."""
        print(f"\n{'=' * 60}")
        print(f"Simulation Results")
        print(f"{'=' * 60}")
        print(f"Total simulated: {results['n_simulated']}")
        print(f"\nOutcomes:")
        print(f"  Correct:       {results['n_correct']:6d} ({results['accuracy']:.2%})")
        print(f"  Near errors:   {results['n_near_error']:6d} ({results['near_error_rate']:.2%})")
        print(f"  Far errors:    {results['n_far_error']:6d} ({results['far_error_rate']:.2%})")
        print(f"  Ambiguous:     {results['n_ambiguous']:6d} ({results['ambiguous_rate']:.2%})")
        print(f"  Failed:        {results['n_failed']:6d} ({results['failed_rate']:.2%})")
        
        if 'x_distance_mean' in results:
            print(f"\nEdit distances (successful decodings):")
            print(f"  X: mean={results['x_distance_mean']:.2f}, max={results['x_distance_max']}")
            print(f"  Y: mean={results['y_distance_mean']:.2f}, max={results['y_distance_max']}")
