#!/usr/bin/env python3
"""
Output utilities for quadtree barcode generation.

Provides column-major export and validation functions for photolithography array fabrication.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def write_barcodes_column_major(df: pd.DataFrame, 
                                out_path: str, 
                                strict: bool = True,
                                rows: Optional[int] = None,
                                cols: Optional[int] = None):
    """
    Write barcodes in column-major order for array fabrication.
    
    Critical for photolithography: array is synthesized column-by-column.
    
    Args:
        df: DataFrame with columns ['row', 'col', 'barcode']
        out_path: Output file path (.txt or .fasta)
        strict: If True, validate complete grid coverage
        rows: Expected number of rows (for validation)
        cols: Expected number of columns (for validation)
    """
    if not {'row', 'col', 'barcode'}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: row, col, barcode")
    
    # Sort column-major: by col first, then by row
    df_sorted = df.sort_values(['col', 'row']).reset_index(drop=True)
    
    # Strict validation
    if strict:
        # Check for duplicates
        duplicates = df_sorted.duplicated(subset=['row', 'col'])
        if duplicates.any():
            dupe_positions = df_sorted[duplicates][['row', 'col']].values
            raise ValueError(f"Duplicate positions found: {dupe_positions[:5]}")
        
        # Check grid coverage if dimensions provided
        if rows is not None and cols is not None:
            expected_count = rows * cols
            actual_count = len(df_sorted)
            if actual_count != expected_count:
                raise ValueError(
                    f"Grid coverage mismatch: expected {expected_count} spots "
                    f"({rows}x{cols}), got {actual_count}"
                )
            
            # Check all positions present
            expected_positions = set(
                (r, c) for r in range(rows) for c in range(cols)
            )
            actual_positions = set(
                (row, col) for row, col in df_sorted[['row', 'col']].values
            )
            missing = expected_positions - actual_positions
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} positions in grid. "
                    f"First few: {list(missing)[:5]}"
                )
    
    # Determine output format
    out_path = Path(out_path)
    
    if out_path.suffix == '.fasta':
        with open(out_path, 'w') as f:
            for idx, row_data in df_sorted.iterrows():
                row, col, barcode = row_data['row'], row_data['col'], row_data['barcode']
                f.write(f">spot_r{row}_c{col}\n{barcode}\n")
    
    elif out_path.suffix == '.txt':
        with open(out_path, 'w') as f:
            for barcode in df_sorted['barcode']:
                f.write(f"{barcode}\n")
    
    else:
        raise ValueError(f"Unsupported output format: {out_path.suffix}. Use .txt or .fasta")
    
    print(f"Column-major barcodes written to {out_path}")
    print(f"  Total spots: {len(df_sorted)}")
    print(f"  Order: column-by-column (col 0, col 1, ..., col {df_sorted['col'].max()})")


def write_layout_csv(df: pd.DataFrame, out_path: str):
    """
    Write standard layout CSV file.
    
    Args:
        df: DataFrame with columns ['row', 'col', 'barcode']
        out_path: Output CSV file path
    """
    if not {'row', 'col', 'barcode'}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: row, col, barcode")
    
    df_sorted = df.sort_values(['row', 'col']).reset_index(drop=True)
    df_sorted.to_csv(out_path, index=False)
    
    print(f"Layout CSV written to {out_path}")
    print(f"  Total spots: {len(df_sorted)}")


def validate_grid_coverage(df: pd.DataFrame, rows: int, cols: int) -> bool:
    """
    Validate complete and unique grid coverage.
    
    Args:
        df: DataFrame with columns ['row', 'col']
        rows: Expected number of rows
        cols: Expected number of columns
    
    Returns:
        True if grid coverage is valid
    
    Raises:
        ValueError if validation fails
    """
    # Check dimensions
    if len(df) != rows * cols:
        raise ValueError(
            f"Count mismatch: expected {rows * cols} spots, got {len(df)}"
        )
    
    # Check duplicates
    if df.duplicated(subset=['row', 'col']).any():
        raise ValueError("Duplicate positions found in grid")
    
    # Check all positions present
    expected_positions = set((r, c) for r in range(rows) for c in range(cols))
    actual_positions = set((row, col) for row, col in df[['row', 'col']].values)
    
    missing = expected_positions - actual_positions
    if missing:
        raise ValueError(f"Missing {len(missing)} positions. First few: {list(missing)[:5]}")
    
    extra = actual_positions - expected_positions
    if extra:
        raise ValueError(f"Extra {len(extra)} positions. First few: {list(extra)[:5]}")
    
    return True


def write_statistics_json(stats: dict, out_path: str):
    """
    Write quality statistics to JSON file.
    
    Args:
        stats: Statistics dictionary from generate_barcode_matrix_rll
        out_path: Output JSON file path
    """
    import json
    
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics written to {out_path}")
    print(f"  Total forced flips: {stats.get('total_forced_flips', 0)}")
    print(f"  Mean forced flip rate: {stats.get('mean_forced_flip_rate', 0):.4f}")
