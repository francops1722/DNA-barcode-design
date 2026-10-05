#!/usr/bin/env python3
"""
Quadtree RLL (Run-Length Limiting) Module

Implements homopolymer prevention for quadtree barcode generation.
Ported from BaseCont_Barcodes.ipynb to provide production-quality
barcode generation with strict run-length limits.
"""

import numpy as np
import itertools
from typing import Tuple, Dict

BASES = np.array(["A", "C", "G", "T"], dtype="<U1")


def _u_v_indices(rows: int, cols: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled row and column indices for quadtree generation.
    
    Args:
        rows: Number of rows in grid
        cols: Number of columns in grid
        length: Barcode length (number of cycles)
    
    Returns:
        Tuple of (u, v) index arrays
    """
    scale = 1 << length  # 2^length
    u = np.floor(np.arange(rows)[:, None] * scale / rows).astype(np.int64)  # (rows, 1)
    v = np.floor(np.arange(cols)[None, :] * scale / cols).astype(np.int64)  # (1, cols)
    return u, v


def _quad_idx_at_level(u: np.ndarray, v: np.ndarray, shift: int) -> np.ndarray:
    """
    Compute quadrant index (0-3) at a specific bit level.
    
    Args:
        u: Row indices array
        v: Column indices array
        shift: Bit shift amount
    
    Returns:
        Array of quadrant indices (0-3)
    """
    row_bit = (u >> shift) & 1
    col_bit = (v >> shift) & 1
    return (row_bit << 1) | col_bit  # values 0..3


def _best_perm_avoid_violations(quad_idx: np.ndarray, 
                                prev_base_idx: np.ndarray, 
                                run_len: np.ndarray, 
                                max_run: int) -> Tuple[np.ndarray, int, int]:
    """
    Select the best permutation of bases to quadrants to minimize homopolymer violations.
    
    Strategy:
    1. Try all 24 possible permutations of (A, C, G, T) -> (quad 0, 1, 2, 3)
    2. Choose permutation that minimizes violations (cells that would exceed max_run)
    3. Use total repeats as tie-breaker
    
    Args:
        quad_idx: Quadrant index (0-3) for each cell
        prev_base_idx: Previous base index (0-3 for A/C/G/T) for each cell
        run_len: Current run length for each cell
        max_run: Maximum allowed run length
    
    Returns:
        Tuple of (best_permutation, violations, repeats)
        - best_permutation: array of length 4 mapping quadrant -> base index
        - violations: number of cells that would violate max_run with this permutation
        - repeats: total number of cells that repeat previous base
    """
    # Build 4x4 contingency: counts[quadrant, prev_base] = number of cells
    combined = (quad_idx.ravel() << 2) | prev_base_idx.ravel()  # 4*quadrant + prev_base
    counts_all = np.bincount(combined, minlength=16).reshape(4, 4)
    
    # Same but only for cells at risk of exceeding the run cap
    at_risk = (run_len.ravel() >= max_run).astype(np.int32)
    counts_risk = np.zeros((4, 4), dtype=np.int64)
    
    for b in range(4):
        for k in range(4):
            mask = (quad_idx.ravel() == b) & (prev_base_idx.ravel() == k)
            if mask.any():
                counts_risk[b, k] = at_risk[mask].sum()
    
    # Try all 24 permutations
    best = None
    best_p = None
    
    for p in itertools.permutations(range(4)):
        # Violations = repeating at at-risk cells
        # If we assign base p[quadrant] to quadrant, cells in quadrant with prev_base=p[quadrant] violate
        viol = counts_risk[np.arange(4), list(p)].sum()
        
        # Repeats overall (used as tie-breaker)
        reps = counts_all[np.arange(4), list(p)].sum()
        
        key = (viol, reps)
        if (best is None) or (key < best):
            best, best_p = key, np.array(p, dtype=np.int8)
    
    viol, reps = best
    return best_p, int(viol), int(reps)


def generate_barcode_matrix_rll(rows: int, 
                                cols: int, 
                                length: int, 
                                max_run: int = 1,
                                verbose: bool = True,
                                initial_permutation=None) -> Tuple[np.ndarray, Dict]:
    """
    Generate quadtree barcodes with run-length limiting (RLL).
    
    This is the production-quality version that prevents homopolymers by:
    1. Dynamically selecting base->quadrant permutations at each cycle
    2. Forcing alternative bases for cells that would exceed max_run
    
    Args:
        rows: Number of rows in grid
        cols: Number of columns in grid
        length: Barcode length in nucleotides
        max_run: Maximum allowed consecutive identical bases (default=1 forbids homopolymers)
        verbose: Print progress updates (default=True)
    
    Returns:
        Tuple of (barcode_matrix, statistics_dict)
        - barcode_matrix: (rows, cols) array of barcode strings
        - statistics_dict: quality metrics including forced flips and violations avoided
    """
    if rows <= 0 or cols <= 0 or length <= 0:
        raise ValueError("rows, cols, and length must be positive integers")
    if max_run < 1:
        raise ValueError("max_run must be >= 1")

    if initial_permutation is not None:
        initial_permutation = np.asarray(initial_permutation, dtype=np.int8)
        if initial_permutation.shape != (4,) or set(initial_permutation.tolist()) != {0, 1, 2, 3}:
            raise ValueError("initial_permutation must be a permutation of [0, 1, 2, 3]")
    
    if verbose and length > 15:
        print(f"    Generating {length} cycles (this may take a minute)...", flush=True)
    
    u, v = _u_v_indices(rows, cols, length)
    chars = np.empty((rows, cols, length), dtype="<U1")
    
    # Book-keeping for RLL
    prev_base_idx = np.zeros((rows, cols), dtype=np.int8)
    run_len = np.zeros((rows, cols), dtype=np.int16)
    forced_flips = []
    viol_counts = []
    rep_counts = []
    
    rows_idx = np.arange(rows)[:, None]
    cols_idx = np.arange(cols)[None, :]
    
    # Cycle 0: use identity permutation (fixed)
    shift0 = length - 1
    q0 = _quad_idx_at_level(u, v, shift0)
    if initial_permutation is None:
        base0 = q0  # 0->A, 1->C, 2->G, 3->T
    else:
        base0 = initial_permutation[q0]
    chars[..., 0] = BASES[base0]
    prev_base_idx = base0.astype(np.int8)
    run_len[:] = 1
    forced_flips.append(0)
    viol_counts.append(0)
    rep_counts.append(0)
    
    # Subsequent cycles
    for l in range(1, length):
        if verbose and length > 15 and l % 5 == 0:
            print(f"      Cycle {l}/{length}...", flush=True)
        
        shift = length - 1 - l
        q = _quad_idx_at_level(u, v, shift)
        
        # Step 1: Choose best permutation to avoid violations
        p, viol, reps = _best_perm_avoid_violations(q, prev_base_idx, run_len, max_run)
        cand = p[q]  # Candidate base index (0..3) before enforcing cap
        
        # Step 2: Enforce the cap - cells that would exceed must flip
        exceed = (cand == prev_base_idx) & (run_len >= max_run)
        
        if np.any(exceed):
            # Deterministic, spatially balanced alternative base
            # offset picks 1, 2, or 3 based on (row + col + cycle) so alt != prev_base_idx
            offset = ((rows_idx + cols_idx + l) % 3) + 1  # values in {1, 2, 3}
            alt = (prev_base_idx + offset) % 4  # 0..3, guaranteed != prev_base_idx
            cand = np.where(exceed, alt, cand)
        
        # Write bases for this cycle
        chars[..., l] = BASES[cand]
        
        # Update run statistics
        same = (cand == prev_base_idx)
        run_len = np.where(same, run_len + 1, 1)
        prev_base_idx = cand.astype(np.int8)
        
        forced_flips.append(int(exceed.sum()))
        viol_counts.append(viol)
        rep_counts.append(reps)
    
    # Join characters into full strings
    barcodes = np.apply_along_axis(lambda x: "".join(x), 2, chars).astype(f"<U{length}")
    
    # Compile statistics
    total_cells = rows * cols
    stats = {
        "forced_flips_per_cycle": forced_flips,
        "violations_avoided_by_perm": viol_counts,
        "pre_flip_repeats_per_cycle": rep_counts,
        "max_run": max_run,
        "mean_forced_flip_rate": (
            float(np.mean(np.array(forced_flips[1:], dtype=float) / total_cells)) 
            if length > 1 else 0.0
        ),
        "total_forced_flips": sum(forced_flips),
        "total_violations_avoided": sum(viol_counts),
    }
    
    return barcodes, stats


def verify_no_homopolymers(barcodes: np.ndarray, max_run: int = 1) -> bool:
    """
    Verify that no barcode exceeds the run-length limit.
    
    Args:
        barcodes: Array of barcode strings
        max_run: Maximum allowed consecutive identical bases
    
    Returns:
        True if all barcodes satisfy the constraint, False otherwise
    """
    flat = barcodes.flatten()
    
    for bc in flat:
        for i in range(len(bc) - max_run):
            window = bc[i:i+max_run+1]
            if len(set(window)) == 1:  # All same base
                return False
    
    return True


def count_homopolymers(barcodes: np.ndarray, min_length: int = 2) -> Dict:
    """
    Count homopolymer occurrences in barcode set.
    
    Args:
        barcodes: Array of barcode strings
        min_length: Minimum homopolymer length to count
    
    Returns:
        Dictionary with homopolymer statistics
    """
    flat = barcodes.flatten()
    homopolymers = {base: {length: 0 for length in range(min_length, 10)} 
                   for base in ['A', 'C', 'G', 'T']}
    
    for bc in flat:
        for i in range(len(bc)):
            base = bc[i]
            run = 1
            for j in range(i+1, len(bc)):
                if bc[j] == base:
                    run += 1
                else:
                    break
            
            if run >= min_length and run < 10:
                homopolymers[base][run] += 1
    
    return homopolymers
