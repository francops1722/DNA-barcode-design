"""
Distance analysis module.

Computes and analyzes distance distributions for axis codes and DNA sequences.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_hamming_distance_matrix(codes: List[List[int]]) -> np.ndarray:
    """
    Compute pairwise Hamming distance matrix for binary codes.
    
    Args:
        codes: List of binary codes
        
    Returns:
        NxN matrix of Hamming distances
    """
    n = len(codes)
    distances = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(b1 != b2 for b1, b2 in zip(codes[i], codes[j]))
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def compute_dna_distance_matrix(sequences: List[str], metric: str = "hamming") -> np.ndarray:
    """
    Compute pairwise distance matrix for DNA sequences.
    
    Args:
        sequences: List of DNA sequences
        metric: Distance metric ("hamming" or "levenshtein")
        
    Returns:
        NxN matrix of distances
    """
    n = len(sequences)
    distances = np.zeros((n, n), dtype=int)
    
    if metric == "hamming":
        for i in range(n):
            for j in range(i + 1, n):
                if len(sequences[i]) != len(sequences[j]):
                    raise ValueError(
                        f"Sequences {i} and {j} have different lengths for Hamming distance"
                    )
                dist = sum(b1 != b2 for b1, b2 in zip(sequences[i], sequences[j]))
                distances[i, j] = dist
                distances[j, i] = dist
    
    elif metric == "levenshtein":
        for i in range(n):
            for j in range(i + 1, n):
                dist = levenshtein_distance(sequences[i], sequences[j])
                distances[i, j] = dist
                distances[j, i] = dist
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    
    Uses dynamic programming.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class AxisDistanceAnalyzer:
    """
    Analyzes distance distributions for 1D axis codes.
    
    Groups distances by spatial separation (position difference) and
    provides summary statistics.
    """
    
    def __init__(self, codes: List[List[int]], axis_name: str = ""):
        """
        Initialize analyzer.
        
        Args:
            codes: List of binary codes for the axis
            axis_name: Name for reporting (e.g., "X" or "Y")
        """
        self.codes = codes
        self.axis_name = axis_name
        self.n_positions = len(codes)
        
        # Compute distance matrix
        self.distance_matrix = compute_hamming_distance_matrix(codes)
        
        # Group by position difference
        self.distances_by_diff: Dict[int, List[int]] = defaultdict(list)
        self._compute_distances_by_diff()
        
        logger.info(f"Initialized distance analyzer for {axis_name} axis ({self.n_positions} codes)")
    
    def _compute_distances_by_diff(self):
        """Group distances by position difference."""
        for i in range(self.n_positions):
            for j in range(i + 1, self.n_positions):
                diff = j - i
                dist = self.distance_matrix[i, j]
                self.distances_by_diff[diff].append(dist)
    
    def get_statistics(self, max_diff: int = None) -> Dict[int, Dict[str, float]]:
        """
        Get summary statistics for each position difference.
        
        Args:
            max_diff: Maximum position difference to include (None = all)
            
        Returns:
            Dictionary mapping position difference to statistics
        """
        stats = {}
        
        for diff in sorted(self.distances_by_diff.keys()):
            if max_diff is not None and diff > max_diff:
                continue
            
            distances = self.distances_by_diff[diff]
            
            stats[diff] = {
                'n_pairs': len(distances),
                'min': int(np.min(distances)),
                'max': int(np.max(distances)),
                'mean': float(np.mean(distances)),
                'median': float(np.median(distances)),
                'std': float(np.std(distances)),
                'q25': float(np.percentile(distances, 25)),
                'q75': float(np.percentile(distances, 75)),
            }
        
        return stats
    
    def get_histogram(self, position_diff: int) -> Dict[int, int]:
        """
        Get histogram of distances for a specific position difference.
        
        Args:
            position_diff: Position difference to analyze
            
        Returns:
            Dictionary mapping distance to count
        """
        if position_diff not in self.distances_by_diff:
            return {}
        
        distances = self.distances_by_diff[position_diff]
        histogram = defaultdict(int)
        
        for dist in distances:
            histogram[dist] += 1
        
        return dict(histogram)
    
    def check_constraints(
        self,
        target_distances: List[int],
        far_features_min_distance: int,
        tolerance: float = 1.0,
    ) -> Dict[str, any]:
        """
        Check if codes satisfy distance constraints.
        
        Args:
            target_distances: Target distances for diffs 1,2,3,4
            far_features_min_distance: Minimum distance for diff >= 5
            tolerance: Tolerance for near-target distances
            
        Returns:
            Dictionary with constraint satisfaction results
        """
        results = {
            'near_constraints': {},
            'far_constraint': {},
            'violations': [],
        }
        
        # Check near constraints (diffs 1-4)
        for diff in range(1, 5):
            if diff not in self.distances_by_diff:
                continue
            
            target = target_distances[diff - 1]
            distances = self.distances_by_diff[diff]
            
            mean_dist = np.mean(distances)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            in_tolerance = sum(
                abs(d - target) <= tolerance for d in distances
            )
            
            results['near_constraints'][diff] = {
                'target': target,
                'mean': mean_dist,
                'min': min_dist,
                'max': max_dist,
                'n_in_tolerance': in_tolerance,
                'fraction_in_tolerance': in_tolerance / len(distances),
            }
        
        # Check far constraint (diff >= 5)
        far_distances = []
        far_violations = 0
        
        for diff in self.distances_by_diff.keys():
            if diff >= 5:
                far_distances.extend(self.distances_by_diff[diff])
                far_violations += sum(
                    d < far_features_min_distance for d in self.distances_by_diff[diff]
                )
        
        if far_distances:
            results['far_constraint'] = {
                'min_required': far_features_min_distance,
                'actual_min': int(np.min(far_distances)),
                'actual_mean': float(np.mean(far_distances)),
                'n_violations': far_violations,
                'n_total': len(far_distances),
                'fraction_violations': far_violations / len(far_distances),
            }
            
            if far_violations > 0:
                results['violations'].append(
                    f"Far constraint: {far_violations}/{len(far_distances)} "
                    f"pairs below minimum distance {far_features_min_distance}"
                )
        
        results['passed'] = len(results['violations']) == 0
        
        return results
    
    def print_summary(
        self,
        target_distances: List[int] = None,
        far_features_min_distance: int = None,
    ):
        """
        Print human-readable summary of distance analysis.
        
        Args:
            target_distances: Optional target distances for comparison
            far_features_min_distance: Optional far-features constraint
        """
        print(f"\n{'=' * 60}")
        print(f"Distance Analysis: {self.axis_name} Axis")
        print(f"{'=' * 60}")
        print(f"Number of positions: {self.n_positions}")
        print(f"Bit length: {len(self.codes[0]) if self.codes else 0}")
        
        # Statistics for position differences 1-5
        print(f"\n{'Position Diff':<15} {'Count':<10} {'Min':<8} {'Mean':<8} {'Max':<8} {'Target':<8}")
        print("-" * 60)
        
        for diff in sorted(self.distances_by_diff.keys())[:10]:
            stats = self.get_statistics()[diff]
            target_str = ""
            
            if target_distances and 1 <= diff <= 4:
                target = target_distances[diff - 1]
                target_str = str(target)
            elif far_features_min_distance and diff >= 5:
                target_str = f">={far_features_min_distance}"
            
            print(
                f"{diff:<15} {stats['n_pairs']:<10} {stats['min']:<8} "
                f"{stats['mean']:<8.1f} {stats['max']:<8} {target_str:<8}"
            )
        
        # Constraint checking
        if target_distances and far_features_min_distance:
            print(f"\n{'Constraint Satisfaction:'}")
            print("-" * 60)
            
            constraint_results = self.check_constraints(
                target_distances, far_features_min_distance
            )
            
            for diff, info in constraint_results['near_constraints'].items():
                status = "✓" if info['fraction_in_tolerance'] > 0.8 else "⚠"
                print(
                    f"{status} Diff {diff}: mean={info['mean']:.1f} "
                    f"(target={info['target']}, in_tolerance={info['fraction_in_tolerance']:.1%})"
                )
            
            if constraint_results['far_constraint']:
                fc = constraint_results['far_constraint']
                status = "✓" if fc['n_violations'] == 0 else "✗"
                print(
                    f"{status} Far (≥5): min={fc['actual_min']} "
                    f"(required≥{fc['min_required']}, violations={fc['n_violations']})"
                )


def analyze_axis_codes(
    x_codes: List[List[int]],
    y_codes: List[List[int]],
    target_distances: List[int],
    far_features_min_distance: int,
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Analyze distance distributions for both X and Y axis codes.
    
    Args:
        x_codes: Binary codes for X axis
        y_codes: Binary codes for Y axis
        target_distances: Target distances for diffs 1-4
        far_features_min_distance: Minimum distance for diff >= 5
        
    Returns:
        Tuple of (x_results, y_results) dictionaries
    """
    logger.info("Analyzing X axis distances...")
    x_analyzer = AxisDistanceAnalyzer(x_codes, "X")
    x_results = x_analyzer.check_constraints(
        target_distances, far_features_min_distance
    )
    x_results['statistics'] = x_analyzer.get_statistics(max_diff=10)
    
    logger.info("Analyzing Y axis distances...")
    y_analyzer = AxisDistanceAnalyzer(y_codes, "Y")
    y_results = y_analyzer.check_constraints(
        target_distances, far_features_min_distance
    )
    y_results['statistics'] = y_analyzer.get_statistics(max_diff=10)
    
    # Print summaries
    x_analyzer.print_summary(target_distances, far_features_min_distance)
    y_analyzer.print_summary(target_distances, far_features_min_distance)
    
    return x_results, y_results
