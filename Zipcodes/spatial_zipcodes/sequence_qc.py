"""
Sequence quality control module.

Implements checks for GC content, G content, homopolymer runs, and
problematic sequence motifs in DNA sequences.

G-content is tracked separately from GC content because in
photolithography-based synthesis G has measurably lower coupling
efficiency.  Barcodes with high G fraction (experimentally > ~25-30%)
produce spots with low read counts.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as fraction (0-1)
    """
    if not sequence:
        return 0.0
    
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def calculate_g_content(sequence: str) -> float:
    """
    Calculate G (guanine) fraction of a DNA sequence.

    Tracked separately from GC content because photolithography synthesis
    has lower coupling efficiency for G, leading to low-read-count spots
    when G fraction exceeds ~25–30%.

    Args:
        sequence: DNA sequence string

    Returns:
        G fraction as fraction (0-1)
    """
    if not sequence:
        return 0.0
    return sequence.upper().count('G') / len(sequence)


def find_max_homopolymer(sequence: str) -> Tuple[int, str]:
    """
    Find the longest homopolymer run in a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Tuple of (max_length, base) where base is the repeated nucleotide
    """
    if not sequence:
        return 0, ''
    
    sequence = sequence.upper()
    max_length = 1
    max_base = sequence[0]
    current_length = 1
    current_base = sequence[0]
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_base:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                max_base = current_base
        else:
            current_base = sequence[i]
            current_length = 1
    
    return max_length, max_base


def find_all_homopolymers(sequence: str, min_length: int = 3) -> List[Dict[str, any]]:
    """
    Find all homopolymer runs above a minimum length.
    
    Args:
        sequence: DNA sequence string
        min_length: Minimum homopolymer length to report
        
    Returns:
        List of dicts with keys: 'base', 'length', 'start', 'end'
    """
    if not sequence:
        return []
    
    sequence = sequence.upper()
    homopolymers = []
    current_base = sequence[0]
    start = 0
    length = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_base:
            length += 1
        else:
            if length >= min_length:
                homopolymers.append({
                    'base': current_base,
                    'length': length,
                    'start': start,
                    'end': i - 1,
                })
            current_base = sequence[i]
            start = i
            length = 1
    
    # Check final run
    if length >= min_length:
        homopolymers.append({
            'base': current_base,
            'length': length,
            'start': start,
            'end': len(sequence) - 1,
        })
    
    return homopolymers


def check_problematic_motifs(sequence: str, motifs: List[str] = None) -> List[Dict[str, any]]:
    """
    Check for problematic sequence motifs.
    
    Args:
        sequence: DNA sequence string
        motifs: List of motif patterns to check (default: common problematic motifs)
        
    Returns:
        List of dicts with keys: 'motif', 'positions'
    """
    if motifs is None:
        # Default problematic motifs
        motifs = [
            'GGGG',   # G quadruplex
            'CCCC',   # C quadruplex
            'AAAAAA', # Long A runs
            'TTTTTT', # Long T runs
            'GCGCGC', # High alternating GC
            'CGCGCG', # High alternating CG
        ]
    
    sequence = sequence.upper()
    found_motifs = []
    
    for motif in motifs:
        motif = motif.upper()
        positions = []
        start = 0
        
        while True:
            pos = sequence.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            found_motifs.append({
                'motif': motif,
                'positions': positions,
                'count': len(positions),
            })
    
    return found_motifs


class SequenceQC:
    """
    Quality control checker for DNA sequences.
    
    Validates sequences against configurable thresholds for GC content,
    homopolymer runs, and problematic motifs.
    """
    
    def __init__(
        self,
        gc_content_min: float = 0.3,
        gc_content_max: float = 0.7,
        g_content_max: float = 1.0,
        max_homopolymer_length: int = 4,
        reject_long_homopolymers: bool = True,
        problematic_motifs: Optional[List[str]] = None,
    ):
        """
        Initialize sequence QC checker.
        
        Args:
            gc_content_min: Minimum acceptable GC content
            gc_content_max: Maximum acceptable GC content
            g_content_max: Maximum acceptable G fraction (default 1.0 = no cap).
                           Set to 0.25 for photolithography arrays to avoid low-read
                           spots caused by poor G coupling efficiency.
            max_homopolymer_length: Maximum acceptable homopolymer length
            reject_long_homopolymers: Whether to reject sequences with long homopolymers
            problematic_motifs: List of motifs to check for
        """
        self.gc_content_min = gc_content_min
        self.gc_content_max = gc_content_max
        self.g_content_max = g_content_max
        self.max_homopolymer_length = max_homopolymer_length
        self.reject_long_homopolymers = reject_long_homopolymers
        self.problematic_motifs = problematic_motifs
        
        logger.info(
            f"Initialized SequenceQC: GC={gc_content_min:.2f}-{gc_content_max:.2f}, "
            f"G<={g_content_max:.2f}, max_homopolymer={max_homopolymer_length}"
        )
    
    def check_sequence(self, sequence: str, sequence_id: str = None) -> Dict[str, any]:
        """
        Perform QC checks on a single sequence.
        
        Args:
            sequence: DNA sequence string
            sequence_id: Optional identifier for reporting
            
        Returns:
            Dictionary with QC results:
                - 'passed': bool
                - 'gc_content': float
                - 'gc_pass': bool
                - 'max_homopolymer_length': int
                - 'max_homopolymer_base': str
                - 'homopolymer_pass': bool
                - 'problematic_motifs': list
                - 'motif_pass': bool
                - 'failures': list of failure messages
        """
        results = {
            'sequence_id': sequence_id,
            'length': len(sequence),
            'failures': [],
        }
        
        # GC content check
        gc = calculate_gc_content(sequence)
        results['gc_content'] = gc
        results['gc_pass'] = self.gc_content_min <= gc <= self.gc_content_max
        
        if not results['gc_pass']:
            results['failures'].append(
                f"GC content {gc:.3f} outside range [{self.gc_content_min:.3f}, {self.gc_content_max:.3f}]"
            )
        
        # G content check (separate from GC: photolithography G coupling efficiency)
        g = calculate_g_content(sequence)
        results['g_content'] = g
        results['g_pass'] = g <= self.g_content_max
        
        if not results['g_pass']:
            results['failures'].append(
                f"G content {g:.3f} exceeds maximum {self.g_content_max:.3f}"
            )
        
        # Homopolymer check
        max_homo_len, max_homo_base = find_max_homopolymer(sequence)
        results['max_homopolymer_length'] = max_homo_len
        results['max_homopolymer_base'] = max_homo_base
        results['homopolymer_pass'] = max_homo_len <= self.max_homopolymer_length
        
        if not results['homopolymer_pass'] and self.reject_long_homopolymers:
            results['failures'].append(
                f"Homopolymer run of {max_homo_base} x {max_homo_len} "
                f"exceeds limit {self.max_homopolymer_length}"
            )
        
        # Problematic motifs check
        motifs_found = check_problematic_motifs(sequence, self.problematic_motifs)
        results['problematic_motifs'] = motifs_found
        results['motif_pass'] = len(motifs_found) == 0
        
        if not results['motif_pass']:
            for motif_info in motifs_found:
                results['failures'].append(
                    f"Found problematic motif '{motif_info['motif']}' "
                    f"at {motif_info['count']} position(s)"
                )
        
        # Overall pass/fail
        results['passed'] = len(results['failures']) == 0
        
        return results
    
    def check_sequences(
        self,
        sequences: List[str],
        sequence_ids: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Perform QC checks on multiple sequences.
        
        Args:
            sequences: List of DNA sequences
            sequence_ids: Optional list of identifiers
            
        Returns:
            Dictionary with aggregate results:
                - 'n_sequences': int
                - 'n_passed': int
                - 'n_failed': int
                - 'pass_rate': float
                - 'individual_results': list of per-sequence results
                - 'failed_sequences': list of failed sequence IDs/indices
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        individual_results = []
        failed_sequences = []
        
        for i, (seq, seq_id) in enumerate(zip(sequences, sequence_ids)):
            result = self.check_sequence(seq, seq_id)
            individual_results.append(result)
            
            if not result['passed']:
                failed_sequences.append(seq_id)
        
        n_passed = sum(1 for r in individual_results if r['passed'])
        n_failed = len(sequences) - n_passed
        
        aggregate = {
            'n_sequences': len(sequences),
            'n_passed': n_passed,
            'n_failed': n_failed,
            'pass_rate': n_passed / len(sequences) if sequences else 0.0,
            'individual_results': individual_results,
            'failed_sequences': failed_sequences,
        }
        
        # Summary statistics
        if sequences:
            gc_contents = [r['gc_content'] for r in individual_results]
            homopolymer_lengths = [r['max_homopolymer_length'] for r in individual_results]
            
            import numpy as np
            aggregate['gc_content_mean'] = np.mean(gc_contents)
            aggregate['gc_content_std'] = np.std(gc_contents)
            aggregate['gc_content_min'] = np.min(gc_contents)
            aggregate['gc_content_max'] = np.max(gc_contents)
            
            aggregate['homopolymer_mean'] = np.mean(homopolymer_lengths)
            aggregate['homopolymer_max'] = np.max(homopolymer_lengths)
        
        logger.info(
            f"QC results: {n_passed}/{len(sequences)} passed ({aggregate['pass_rate']:.1%})"
        )
        
        return aggregate


def validate_axis_sequences(
    sequences: List[str],
    axis_name: str,
    qc_config: Dict[str, any],
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate a set of axis sequences (X or Y codes).
    
    Args:
        sequences: List of DNA sequences
        axis_name: Name of axis for reporting ("X" or "Y")
        qc_config: QC configuration dict
        
    Returns:
        Tuple of (all_passed, results_dict)
    """
    qc = SequenceQC(
        gc_content_min=qc_config.get('gc_content_min', 0.3),
        gc_content_max=qc_config.get('gc_content_max', 0.7),
        g_content_max=qc_config.get('g_content_max', 1.0),
        max_homopolymer_length=qc_config.get('max_homopolymer_length', 4),
        reject_long_homopolymers=qc_config.get('reject_sequences_with_long_homopolymers', True),
    )
    
    sequence_ids = [f"{axis_name}_{i}" for i in range(len(sequences))]
    results = qc.check_sequences(sequences, sequence_ids)
    
    all_passed = results['n_failed'] == 0
    
    if not all_passed:
        logger.warning(
            f"{axis_name} axis: {results['n_failed']}/{results['n_sequences']} "
            f"sequences failed QC"
        )
        # Log first few failures
        for result in results['individual_results'][:5]:
            if not result['passed']:
                logger.warning(
                    f"  {result['sequence_id']}: {', '.join(result['failures'])}"
                )
    else:
        logger.info(f"{axis_name} axis: all {results['n_sequences']} sequences passed QC")
    
    return all_passed, results
