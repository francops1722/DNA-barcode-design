"""
Axis code generator for spatial zipcodes.

Implements a randomized greedy algorithm with backtracking to generate
1D sequences of binary codes that satisfy distance constraints.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def hamming_distance(code1: List[int], code2: List[int]) -> int:
    """
    Calculate Hamming distance between two binary codes.
    
    Args:
        code1: First binary code (list of 0s and 1s)
        code2: Second binary code (list of 0s and 1s)
        
    Returns:
        Hamming distance (number of differing bits)
    """
    if len(code1) != len(code2):
        raise ValueError(f"Codes must have same length: {len(code1)} vs {len(code2)}")
    return sum(b1 != b2 for b1, b2 in zip(code1, code2))


def random_binary_code(length: int, rng: random.Random) -> List[int]:
    """
    Generate a random binary code of specified length.
    
    Args:
        length: Number of bits
        rng: Random number generator
        
    Returns:
        List of 0s and 1s
    """
    return [rng.randint(0, 1) for _ in range(length)]


class AxisCodeGenerator:
    """
    Generates 1D axis codes with specified distance constraints.
    
    Uses a randomized greedy algorithm with backtracking to construct
    codes that satisfy hard constraints (far-features minimum distance)
    and approximate soft constraints (target distances for nearby positions).
    """
    
    def __init__(
        self,
        n_positions: int,
        bit_length: int,
        target_distances: List[int],
        far_features_min_distance: int,
        max_attempts_per_position: int = 10000,
        backtrack_depth: int = 5,
        tolerance: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the axis code generator.
        
        Args:
            n_positions: Number of positions along the axis
            bit_length: Length of binary codes
            target_distances: Target Hamming distances for position differences 1,2,3,4
            far_features_min_distance: Minimum distance for positions >= 5 apart
            max_attempts_per_position: Max random samples per position before backtracking
            backtrack_depth: Number of positions to backtrack when stuck
            tolerance: Tolerance for soft constraint violations (fraction)
            random_seed: Random seed for reproducibility
        """
        self.n_positions = n_positions
        self.bit_length = bit_length
        self.target_distances = target_distances
        self.far_features_min_distance = far_features_min_distance
        self.max_attempts = max_attempts_per_position
        self.backtrack_depth = backtrack_depth
        self.tolerance = tolerance
        
        self.rng = random.Random(random_seed)
        self.codes: List[List[int]] = []
        
        logger.info(
            f"Initialized AxisCodeGenerator: n={n_positions}, L={bit_length}, "
            f"targets={target_distances}, far_min={far_features_min_distance}"
        )
    
    def _target_distance_for_diff(self, diff: int) -> int:
        """Get target distance for a position difference."""
        if diff == 0:
            return 0
        elif 1 <= diff <= 4:
            return self.target_distances[diff - 1]
        else:  # diff >= 5
            return self.far_features_min_distance
    
    def _check_hard_constraints(self, candidate: List[int], position: int) -> bool:
        """
        Check if candidate satisfies hard constraints.
        
        Hard constraint: minimum distance to all far-away positions (diff >= 5).
        
        Args:
            candidate: Candidate binary code
            position: Current position index
            
        Returns:
            True if all hard constraints satisfied
        """
        for prev_pos in range(position):
            diff = position - prev_pos
            if diff >= 5:
                dist = hamming_distance(candidate, self.codes[prev_pos])
                if dist < self.far_features_min_distance:
                    return False
        return True
    
    def _score_soft_constraints(self, candidate: List[int], position: int) -> float:
        """
        Score candidate based on soft constraints.
        
        Soft constraints: target distances for nearby positions (diff 1-4).
        Lower score is better.
        
        Args:
            candidate: Candidate binary code
            position: Current position index
            
        Returns:
            Score (sum of squared errors from target distances)
        """
        score = 0.0
        for prev_pos in range(position):
            diff = position - prev_pos
            if 1 <= diff <= 4:
                target = self.target_distances[diff - 1]
                actual = hamming_distance(candidate, self.codes[prev_pos])
                error = actual - target
                score += error ** 2
        return score
    
    def _is_acceptable(self, candidate: List[int], position: int) -> bool:
        """
        Check if candidate is acceptable (satisfies hard + soft constraints).
        
        Args:
            candidate: Candidate binary code
            position: Current position index
            
        Returns:
            True if acceptable
        """
        # Must satisfy hard constraints
        if not self._check_hard_constraints(candidate, position):
            return False
        
        # Check soft constraints within tolerance
        score = self._score_soft_constraints(candidate, position)
        
        # Calculate acceptable threshold based on position and tolerance
        # More lenient for positions with more constraints
        n_soft_constraints = min(position, 4)
        if n_soft_constraints == 0:
            return True
        
        # Average squared error per constraint
        avg_sq_error = score / n_soft_constraints
        
        # Accept if average error is within tolerance
        max_acceptable_error = (self.tolerance * max(self.target_distances)) ** 2
        
        return avg_sq_error <= max_acceptable_error
    
    def _generate_position(self, position: int) -> Optional[List[int]]:
        """
        Generate code for a single position.
        
        Args:
            position: Position index
            
        Returns:
            Binary code if successful, None if failed
        """
        best_candidate = None
        best_score = float('inf')
        
        for attempt in range(self.max_attempts):
            candidate = random_binary_code(self.bit_length, self.rng)
            
            if self._check_hard_constraints(candidate, position):
                score = self._score_soft_constraints(candidate, position)
                
                if score < best_score:
                    best_candidate = candidate
                    best_score = score
                
                # Accept if good enough
                if self._is_acceptable(candidate, position):
                    return candidate
        
        # Return best found, even if not ideal
        if best_candidate is not None:
            logger.debug(
                f"Position {position}: accepting best candidate with score {best_score:.2f} "
                f"after {self.max_attempts} attempts"
            )
        return best_candidate
    
    def generate(self) -> List[List[int]]:
        """
        Generate axis codes for all positions.
        
        Uses greedy generation with backtracking when stuck.
        
        Returns:
            List of binary codes (one per position)
            
        Raises:
            RuntimeError: If unable to generate valid codes
        """
        self.codes = []
        position = 0
        backtrack_count = 0
        max_backtracks = 100
        
        logger.info(f"Starting axis code generation for {self.n_positions} positions...")
        
        while position < self.n_positions:
            if position % 10 == 0 and position > 0:
                logger.info(f"Generated codes for {position}/{self.n_positions} positions")
            
            candidate = self._generate_position(position)
            
            if candidate is not None:
                self.codes.append(candidate)
                position += 1
            else:
                # Backtrack
                backtrack_count += 1
                if backtrack_count > max_backtracks:
                    raise RuntimeError(
                        f"Failed to generate codes: exceeded {max_backtracks} backtracks"
                    )
                
                backtrack_amount = min(self.backtrack_depth, position)
                if backtrack_amount == 0:
                    raise RuntimeError(
                        f"Failed to generate code for position 0 "
                        f"(bit_length={self.bit_length} may be too small)"
                    )
                
                logger.warning(
                    f"Backtracking {backtrack_amount} positions from position {position}"
                )
                self.codes = self.codes[:-backtrack_amount]
                position -= backtrack_amount
        
        logger.info(
            f"Successfully generated {len(self.codes)} codes "
            f"(backtracks: {backtrack_count})"
        )
        
        return self.codes
    
    def verify_constraints(self) -> Dict[str, any]:
        """
        Verify generated codes satisfy constraints and compute statistics.
        
        Returns:
            Dictionary with verification results and statistics
        """
        if not self.codes:
            raise ValueError("No codes generated yet")
        
        results = {
            "n_positions": len(self.codes),
            "bit_length": self.bit_length,
            "hard_constraint_violations": 0,
            "soft_constraint_stats": {},
            "distance_distribution": {},
        }
        
        # Check all pairwise distances
        for i in range(len(self.codes)):
            for j in range(i + 1, len(self.codes)):
                diff = j - i
                dist = hamming_distance(self.codes[i], self.codes[j])
                
                # Record in distribution
                if diff not in results["distance_distribution"]:
                    results["distance_distribution"][diff] = []
                results["distance_distribution"][diff].append(dist)
                
                # Check hard constraint
                if diff >= 5 and dist < self.far_features_min_distance:
                    results["hard_constraint_violations"] += 1
                
                # Check soft constraints
                if 1 <= diff <= 4:
                    target = self.target_distances[diff - 1]
                    error = dist - target
                    
                    if diff not in results["soft_constraint_stats"]:
                        results["soft_constraint_stats"][diff] = {
                            "target": target,
                            "errors": [],
                        }
                    results["soft_constraint_stats"][diff]["errors"].append(error)
        
        # Compute summary statistics
        for diff, stats in results["soft_constraint_stats"].items():
            errors = stats["errors"]
            stats["mean_error"] = np.mean(errors)
            stats["std_error"] = np.std(errors)
            stats["mean_abs_error"] = np.mean(np.abs(errors))
            stats["actual_mean_distance"] = stats["target"] + stats["mean_error"]
        
        # Summary of distance distribution
        for diff, distances in results["distance_distribution"].items():
            results["distance_distribution"][diff] = {
                "min": min(distances),
                "max": max(distances),
                "mean": np.mean(distances),
                "std": np.std(distances),
            }
        
        return results


def generate_axis_codes(
    n_positions: int,
    bit_length: int,
    target_distances: List[int],
    far_features_min_distance: int,
    max_attempts_per_position: int = 10000,
    backtrack_depth: int = 5,
    tolerance: float = 0.5,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[int]], Dict[str, any]]:
    """
    Generate and verify axis codes.
    
    Convenience function that creates a generator, generates codes,
    and verifies constraints.
    
    Args:
        n_positions: Number of positions along the axis
        bit_length: Length of binary codes
        target_distances: Target Hamming distances for position differences 1,2,3,4
        far_features_min_distance: Minimum distance for positions >= 5 apart
        max_attempts_per_position: Max random samples per position
        backtrack_depth: Number of positions to backtrack when stuck
        tolerance: Tolerance for soft constraint violations
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (codes, verification_results)
    """
    generator = AxisCodeGenerator(
        n_positions=n_positions,
        bit_length=bit_length,
        target_distances=target_distances,
        far_features_min_distance=far_features_min_distance,
        max_attempts_per_position=max_attempts_per_position,
        backtrack_depth=backtrack_depth,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    
    codes = generator.generate()
    verification = generator.verify_constraints()
    
    return codes, verification
