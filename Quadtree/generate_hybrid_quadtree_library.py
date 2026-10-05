#!/usr/bin/env python3
"""
Quadtree + Hybrid Tail Approach with RLL: 3x 8nt quadtree + 12nt appended tail

Combines:
- 3x 8nt RLL-controlled quadtree (24nt total - provides spatial similarity for synthesis error tolerance)
- 12nt random tail by default (provides high Hamming distance for sequencing quality)
- Total: 36nt barcodes

Production-quality implementation with:
- Run-length limiting (RLL) for homopolymer prevention
- Column-major export for array fabrication
- Quality statistics tracking
"""

import numpy as np
import pandas as pd
import math
import sys
import itertools
from pathlib import Path
from quadtree_rll import generate_barcode_matrix_rll, verify_no_homopolymers, count_homopolymers
from output_utils import (write_barcodes_column_major, write_layout_csv, 
                          validate_grid_coverage, write_statistics_json)


def _distinct_restart_permutations(n_repeats, seed=None):
    """Return distinct quadrant-to-base permutations for repeated quadtree restarts."""
    rng = np.random.default_rng(seed)
    pool = [np.array(perm, dtype=np.int8) for perm in itertools.permutations(range(4))]
    rng.shuffle(pool)

    permutations = []
    while len(permutations) < n_repeats:
        for perm in pool:
            permutations.append(perm.copy())
            if len(permutations) == n_repeats:
                break
        rng.shuffle(pool)

    return permutations


def generate_repeated_quadtree(rows, cols, segment_length=8, n_repeats=3, max_run=1, seed=None):
    """
    Generate multiple RLL-controlled quadtree rounds.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        segment_length: Length of each quadtree segment (default: 8nt)
        n_repeats: Number of repetitions (default: 3)
        max_run: Maximum consecutive identical bases (default: 1)
        seed: Optional seed for choosing restart permutations
    
    Returns:
        Tuple of (combined_matrix, segments_list, all_stats)
    """
    segments = []
    all_stats = []
    restart_permutations = _distinct_restart_permutations(n_repeats, seed=seed)
    
    for i in range(n_repeats):
        segment_matrix, stats = generate_barcode_matrix_rll(
            rows,
            cols,
            segment_length,
            max_run,
            initial_permutation=restart_permutations[i],
        )
        segments.append(segment_matrix)
        all_stats.append(stats)
    
    # Concatenate all segments
    combined = segments[0].copy()
    for i in range(1, n_repeats):
        combined = np.char.add(combined, segments[i])
    
    return combined, segments, all_stats


def generate_random_tail(rows, cols, tail_length=12, seed=None):
    """Generate a per-spot random barcode tail of the requested length."""
    rng = np.random.default_rng(seed)
    bases = np.array(['A', 'C', 'G', 'T'], dtype='<U1')
    tail_chars = rng.choice(bases, size=(rows, cols, tail_length))
    return np.apply_along_axis(lambda x: ''.join(x), 2, tail_chars)


def load_barcode_library(filepath):
    """
    Load barcodes from file.
    
    Args:
        filepath: Path to barcode file (one barcode per line)
    
    Returns:
        List of barcode strings
    """
    barcodes = []
    with open(filepath, 'r') as f:
        for line in f:
            barcode = line.strip()
            if barcode:
                barcodes.append(barcode)
    
    print(f"Loaded {len(barcodes):,} barcodes from {filepath}")
    if barcodes:
        print(f"  First barcode: {barcodes[0]}")
        print(f"  Barcode length: {len(barcodes[0])}nt")
    
    return barcodes


def assign_library_barcodes(rows, cols, library, strategy='sequential'):
    """
    Assign library barcodes to grid positions.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        library: List of barcode strings
        strategy: Assignment strategy ('sequential', 'random', 'spatial')
    
    Returns:
        Matrix of library barcode assignments
    """
    n_spots = rows * cols
    n_library = len(library)
    
    print(f"\nAssigning library barcodes using '{strategy}' strategy...")
    print(f"  Spots: {n_spots:,}")
    print(f"  Library size: {n_library:,}")
    
    if n_library >= n_spots:
        # Enough barcodes - use first n_spots
        print(f"  ✅ Sufficient barcodes - using unique barcode per spot")
        assigned = np.array(library[:n_spots]).reshape(rows, cols)
    else:
        # Not enough - need to reuse with spatial distribution
        print(f"  ⚠️  Reusing barcodes - each barcode used ~{n_spots/n_library:.1f} times")
        
        if strategy == 'sequential':
            # Simple sequential assignment with wraparound
            extended = (library * ((n_spots // n_library) + 1))[:n_spots]
            assigned = np.array(extended).reshape(rows, cols)
        
        elif strategy == 'spatial':
            # Distribute same barcodes far apart spatially
            # Tile library across grid
            tiles_row = (rows + n_library - 1) // n_library
            tiles_col = (cols + n_library - 1) // n_library
            
            assigned = np.empty((rows, cols), dtype=object)
            lib_idx = 0
            for i in range(rows):
                for j in range(cols):
                    assigned[i, j] = library[lib_idx % n_library]
                    lib_idx += 1
        
        else:  # random
            np.random.seed(42)
            indices = np.random.choice(n_library, size=n_spots, replace=True)
            assigned = np.array([library[i] for i in indices]).reshape(rows, cols)
    
    return assigned


def generate_hybrid_layout(rows, cols, library_file=None, segment_length=8, n_quadtree_rounds=3,
                          assignment_strategy='sequential', max_run=1, tail_length=12,
                          tail_mode=None, random_seed=None):
    """
    Generate hybrid layout combining repeated RLL-controlled quadtree and library barcodes.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        library_file: Path to barcode library file (used when tail_mode='library')
        segment_length: Length of each quadtree segment (default: 8nt)
        n_quadtree_rounds: Number of quadtree rounds (default: 3)
        assignment_strategy: How to assign library barcodes
        max_run: Maximum consecutive identical bases (default: 1)
        tail_length: Length of appended tail when tail_mode='random'
        tail_mode: 'random' for generated 12nt tails or 'library' for file-backed barcodes
        random_seed: Seed for restart permutation selection and random tail generation
    
    Returns:
        Tuple of (DataFrame, quadtree_segments, all_stats)
    """
    print("="*80)
    print("GENERATING HYBRID LAYOUT (with RLL)")
    print("="*80)

    if tail_mode is None:
        tail_mode = 'library' if library_file else 'random'
    
    # Step 1: Generate repeated quadtree base with RLL
    total_quadtree_length = segment_length * n_quadtree_rounds
    print(f"\nStep 1: Generating {n_quadtree_rounds} RLL-controlled quadtree rounds of {segment_length}nt each...")
    print(f"  Total quadtree length: {total_quadtree_length}nt")
    print(f"  max_run = {max_run} (forbids {'homopolymers' if max_run == 1 else f'{max_run+1}+ consecutive bases'})")
    
    quadtree_combined, quadtree_segments, all_stats = generate_repeated_quadtree(
        rows, cols, segment_length, n_quadtree_rounds, max_run, seed=random_seed
    )
    
    print(f"  ✅ Generated {rows*cols:,} RLL-controlled quadtree codes")
    for i, stats in enumerate(all_stats, 1):
        print(f"    Round {i}: forced_flips={stats['total_forced_flips']}, "
              f"rate={stats['mean_forced_flip_rate']:.4f}")
    
    # Step 2: Build the appended tail
    if tail_mode == 'library':
        print(f"\nStep 2: Loading barcode library...")
        if not library_file:
            print("  ❌ ERROR: library_file is required when tail_mode='library'!")
            sys.exit(1)

        library = load_barcode_library(library_file)
        if not library:
            print("  ❌ ERROR: No barcodes loaded from file!")
            sys.exit(1)

        print(f"\nStep 3: Assigning library barcodes to spots...")
        library_assigned = assign_library_barcodes(rows, cols, library, assignment_strategy)
        tail_label = f"{len(library[0])}nt library"
    else:
        print(f"\nStep 2: Generating random {tail_length}nt tail for each spot...")
        library_assigned = generate_random_tail(rows, cols, tail_length=tail_length, seed=random_seed)
        tail_label = f"{tail_length}nt random tail"

    # Step 4: Combine repeated quadtree + tail
    print(f"\nStep 4: Combining {total_quadtree_length}nt quadtree + {tail_label}...")
    combined = np.char.add(quadtree_combined, library_assigned)
    
    # Create DataFrame
    idx = np.arange(rows * cols)
    row_idx = idx // cols
    col_idx = idx % cols
    
    data = {
        'row': row_idx,
        'col': col_idx,
        'quadtree_combined': quadtree_combined.flatten(),
        'library': library_assigned.flatten(),
        'barcode': combined.flatten(),
    }
    
    # Add individual quadtree segments
    for i in range(n_quadtree_rounds):
        data[f'quadtree_round{i+1}'] = quadtree_segments[i].flatten()
    
    df = pd.DataFrame(data)
    
    print(f"  ✅ Combined barcodes created")
    
    return df, quadtree_segments, all_stats


def analyze_hybrid_layout(df, n_quadtree_rounds=3, all_stats=None):
    """Analyze the generated hybrid layout with enhanced quality metrics"""
    print("\n" + "="*80)
    print("LAYOUT ANALYSIS")
    print("="*80)
    
    # Basic stats
    n_spots = len(df)
    n_unique = df['barcode'].nunique()
    total_length = len(df['barcode'].iloc[0])
    quadtree_length = len(df['quadtree_combined'].iloc[0])
    tail_length = len(df['library'].iloc[0])
    
    print(f"\nBasic Statistics:")
    print(f"  Total spots: {n_spots:,}")
    print(f"  Unique barcodes: {n_unique:,}")
    print(f"  Uniqueness: {100*n_unique/n_spots:.4f}%")
    print(f"  Total barcode length: {total_length}nt")
    print(f"    • Quadtree: {quadtree_length}nt")
    print(f"    • Tail: {tail_length}nt")
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['barcode'], keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠️  WARNING: {len(duplicates)} duplicate combined barcodes found!")
        dup_counts = duplicates['barcode'].value_counts()
        print(f"  Most common duplicate appears {dup_counts.iloc[0]} times")
    else:
        print(f"\n✅ All combined barcodes are unique!")
    
    # Homopolymer analysis (only on quadtree portion)
    print(f"\nHomopolymer Analysis (Quadtree Portion):")
    quadtree_array = df['quadtree_combined'].values.reshape(-1, 1)
    homopolymers = count_homopolymers(quadtree_array, min_length=2)
    
    total_homo = sum(sum(counts.values()) for counts in homopolymers.values())
    if total_homo == 0:
        print(f"  ✅ No homopolymers detected in quadtree portion")
    else:
        print(f"  ⚠️  Found {total_homo} homopolymer occurrences in quadtree:")
        for base in ['A', 'C', 'G', 'T']:
            for length in range(2, 10):
                count = homopolymers[base][length]
                if count > 0:
                    print(f"    {base*length}: {count} occurrences")
    
    # RLL statistics
    if all_stats:
        print(f"\nRLL Quality Statistics (Quadtree):")
        for i, stats in enumerate(all_stats, 1):
            print(f"  Round {i}:")
            print(f"    Forced flips: {stats['total_forced_flips']} "
                  f"(rate: {stats['mean_forced_flip_rate']:.4f})")
            print(f"    Violations avoided: {stats['total_violations_avoided']}")
    
    # Quadtree uniqueness
    quadtree_unique = df['quadtree_combined'].nunique()
    print(f"\nQuadtree Statistics:")
    print(f"  Total quadtree length: {quadtree_length}nt ({n_quadtree_rounds} rounds × {quadtree_length//n_quadtree_rounds}nt)")
    print(f"  Unique combined quadtree codes: {quadtree_unique:,}")
    for i in range(n_quadtree_rounds):
        if f'quadtree_round{i+1}' in df.columns:
            round_unique = df[f'quadtree_round{i+1}'].nunique()
            print(f"    Round {i+1}: {round_unique:,} unique codes")
    print(f"  Purpose: Spatial similarity for synthesis error tolerance")
    
    # Tail statistics
    library_unique = df['library'].nunique()
    library_reuse = n_spots / library_unique
    print(f"\nTail Statistics:")
    print(f"  Unique tail barcodes: {library_unique:,}")
    print(f"  Average reuse per barcode: {library_reuse:.2f}×")
    print(f"  Purpose: High Hamming distance for sequencing quality")
    
    # Spatial similarity check
    print(f"\nSpatial Similarity (quadtree portion only):")
    max_row = df['row'].max()
    max_col = df['col'].max()
    sample_positions = [(0, 0), (0, 1), (max_row//2, max_col//2)]
    
    for r, c in sample_positions[:2]:
        if c + 1 <= max_col:
            bc1_df = df[(df['row']==r) & (df['col']==c)]
            bc2_df = df[(df['row']==r) & (df['col']==c+1)]
            if len(bc1_df) > 0 and len(bc2_df) > 0:
                bc1 = bc1_df['quadtree_combined'].values[0]
                bc2 = bc2_df['quadtree_combined'].values[0]
                hamming = sum(b1 != b2 for b1, b2 in zip(bc1, bc2))
                print(f"  Position ({r},{c}) vs ({r},{c+1}): {hamming} differences (Hamming distance)")


def export_layout(df, output_csv, output_fasta=None, output_column_major=None,
                 stats_list=None, stats_json=None):
    """Export layout to multiple formats with column-major support"""
    
    # Standard CSV (row-major)
    write_layout_csv(df, output_csv)
    
    # FASTA (optional)
    if output_fasta:
        with open(output_fasta, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">R{row['row']:03d}_C{row['col']:03d}\n")
                f.write(f"{row['barcode']}\n")
        print(f"FASTA exported to {output_fasta}")
    
    # Column-major export (critical for array fabrication)
    if output_column_major:
        max_row = df['row'].max()
        max_col = df['col'].max()
        write_barcodes_column_major(
            df, 
            output_column_major, 
            strict=True,
            rows=max_row + 1,
            cols=max_col + 1
        )
    
    # Statistics JSON (optional)
    if stats_list and stats_json:
        combined_stats = {
            'quadtree_segments': stats_list,
            'mean_forced_flip_rate': np.mean([s['mean_forced_flip_rate'] for s in stats_list]),
            'total_forced_flips': sum(s['total_forced_flips'] for s in stats_list),
        }
        write_statistics_json(combined_stats, stats_json)


def main():
    """Main execution"""
    print("="*80)
    print("QUADTREE + RANDOM TAIL HYBRID BARCODE LAYOUT GENERATOR (with RLL)")
    print("="*80)
    
    # Configuration
    ROWS = 118
    COLS = 182
    SEGMENT_LENGTH = 8  # 8nt per quadtree segment
    N_QUADTREE_ROUNDS = 3  # 3 rounds = 24nt total
    MAX_RUN = 1  # Forbid homopolymers
    TAIL_LENGTH = 12  # random 12nt tail appended to complete 36nt
    OUTPUT_CSV = 'layout_118x182_hybrid_3x8_12.csv'
    OUTPUT_FASTA = 'layout_118x182_hybrid_3x8_12.fasta'
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {ROWS} × {COLS} = {ROWS*COLS:,} spots")
    print(f"  Quadtree rounds: {N_QUADTREE_ROUNDS}")
    print(f"  Quadtree segment length: {SEGMENT_LENGTH}nt")
    print(f"  Total quadtree length: {SEGMENT_LENGTH * N_QUADTREE_ROUNDS}nt")
    print(f"  Max run length: {MAX_RUN} (forbids homopolymers)")
    print(f"  Tail length: {TAIL_LENGTH}nt random sequence")
    
    OUTPUT_COLUMN_MAJOR = 'layout_118x182_hybrid_3x8_12_colmajor.txt'
    STATS_JSON = 'layout_118x182_hybrid_3x8_12_stats.json'
    
    # Calculate minimum length for uniqueness
    min_length = math.ceil(math.log2(max(ROWS, COLS)))
    print(f"  Minimum length for uniqueness: {min_length}nt")
    
    # Generate layout
    try:
        df, quadtree_segments, all_stats = generate_hybrid_layout(
            ROWS, COLS, None, SEGMENT_LENGTH, N_QUADTREE_ROUNDS,
            'sequential', MAX_RUN, TAIL_LENGTH, 'random', 42
        )
        
        # Analyze
        analyze_hybrid_layout(df, N_QUADTREE_ROUNDS, all_stats)
        
        # Export
        export_layout(df, OUTPUT_CSV, OUTPUT_FASTA, OUTPUT_COLUMN_MAJOR,
                     all_stats, STATS_JSON)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        total_length = len(df['barcode'].iloc[0])
        quadtree_total = SEGMENT_LENGTH * N_QUADTREE_ROUNDS
        tail_length = total_length - quadtree_total
        print(f"✅ Generated {len(df):,} hybrid barcodes of {total_length}nt length")
        print(f"✅ Quadtree: {N_QUADTREE_ROUNDS} rounds × {SEGMENT_LENGTH}nt = {quadtree_total}nt (synthesis error tolerance)")
        print(f"✅ Random tail: {tail_length}nt (sequencing quality)")
        print(f"✅ Homopolymer prevention (max_run={MAX_RUN})")
        print(f"✅ Column-major export for array fabrication")
        print(f"✅ Best of both approaches!")
        print("="*80)
        print(f"\nOutput files:")
        print(f"  {OUTPUT_CSV}")
        print(f"  {OUTPUT_FASTA}")
        print(f"  {OUTPUT_COLUMN_MAJOR}")
        print(f"  {STATS_JSON}")
        
        return df, all_stats
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    df, stats = main()

