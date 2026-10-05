#!/usr/bin/env python3
"""
Generate Dual Barcode Layouts from a JSON configuration.

This script supports arbitrary array dimensions and block sizes through
config + CLI overrides, so users can generate new layouts without
editing Python code.

Usage examples:
    python generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
    python generate_dual_layout.py --config my_config.json --rows 253 --cols 363
"""

import sys
import os
import json
import argparse
from math import ceil
from datetime import datetime

# Add parent directory to path to import dual_barcode_layout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_barcode_layout import DualBarcodeLayout, load_barcodes_from_file


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def _resolve_path(config_file, candidate_path):
    """Resolve a possibly-relative path against the config directory."""
    if os.path.isabs(candidate_path):
        return candidate_path
    base_dir = os.path.dirname(os.path.abspath(config_file))
    return os.path.normpath(os.path.join(base_dir, candidate_path))


def _build_default_output_paths(config_file, rows, cols):
    """Build default output names when config omits explicit output fields."""
    base_dir = os.path.dirname(os.path.abspath(config_file))
    stem = f"layout_{rows}x{cols}_dual"
    return {
        "layout_file": os.path.join(base_dir, f"{stem}.csv"),
        "barcode_file": os.path.join(base_dir, f"barcodes_{rows}x{cols}_dual.txt"),
        "stats_file": os.path.join(base_dir, f"stats_{rows}x{cols}_dual.json"),
    }


def generate_layout(
    config_file,
    rows_override=None,
    cols_override=None,
    seed=1,
    output_prefix=None,
):
    """Generate a dual-barcode layout from one config file."""
    print(f"\n{'='*80}")
    print("GENERATING DUAL BARCODE LAYOUT")
    print(f"{'='*80}\n")
    print(f"Configuration: {config_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"[1/6] Loading configuration...")
    config = load_config(config_file)

    rows = int(rows_override) if rows_override is not None else int(config['array']['rows'])
    cols = int(cols_override) if cols_override is not None else int(config['array']['cols'])
    block_rows = int(config.get('block', {}).get('rows', 4))
    block_cols = int(config.get('block', {}).get('cols', 4))

    bar_1_file = _resolve_path(config_file, config['barcodes']['bar_1_file'])
    bar_2_file = _resolve_path(config_file, config['barcodes']['bar_2_file'])

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid array dimensions: rows={rows}, cols={cols}")
    if block_rows != 4 or block_cols != 4:
        raise ValueError(
            "Current DualBarcodeLayout implementation requires 4x4 blocks. "
            f"Received {block_rows}x{block_cols}."
        )
    
    print(f"  Array: {rows}×{cols} = {rows*cols:,} spots")
    print(f"  Block size: {block_rows}×{block_cols} = {block_rows * block_cols} spots/block")
    print(f"  bar_1 library: {bar_1_file}")
    print(f"  bar_2 library: {bar_2_file}")
    
    # Load barcodes
    print(f"\n[2/6] Loading barcode libraries...")
    bar_1 = load_barcodes_from_file(bar_1_file)
    bar_2 = load_barcodes_from_file(bar_2_file)
    
    print(f"  bar_1: {len(bar_1):,} barcodes loaded")
    print(f"  bar_2: {len(bar_2):,} barcodes loaded")
    
    # Calculate requirements
    n_blocks = ceil(rows / block_rows) * ceil(cols / block_cols)
    
    print(f"\n[3/6] Calculating requirements...")
    print(f"  Total blocks needed: {n_blocks:,}")
    print(f"  bar_1 needed: {n_blocks:,}")
    print(f"  bar_2 needed per block: {block_rows * block_cols}")
    
    if len(bar_1) < n_blocks:
        raise ValueError(f"Insufficient bar_1: need {n_blocks}, have {len(bar_1)}")
    if len(bar_2) < (block_rows * block_cols):
        raise ValueError(
            f"Insufficient bar_2: need at least {block_rows * block_cols}, have {len(bar_2)}"
        )
    
    # Generate layout
    print(f"\n[4/6] Generating dual barcode layout...")
    # DualBarcodeLayout uses a single barcode list and generates 4x4 blocks internally
    # We use bar_1 as the barcode source
    layout = DualBarcodeLayout(rows=rows, cols=cols, barcodes=bar_1, barcodes_bar2=bar_2)
    grid_bar1, grid_bar2, grid_combined = layout.generate_layout(seed=seed)
    
    # Flatten to get column-major ordered list
    barcodes_list = []
    for col in range(cols):
        for row in range(rows):
            barcodes_list.append(grid_combined[row, col])
    
    print(f"  ✓ Layout generated")
    print(f"  Total barcodes: {len(barcodes_list):,}")
    print(f"  Unique barcodes: {len(set(barcodes_list)):,}")
    
    # Verify
    print(f"\n[5/6] Verifying layout...")
    stats = layout.get_statistics()
    
    n_unique = len(set(barcodes_list))
    n_total = len(barcodes_list)
    
    if n_unique == n_total:
        print(f"  ✓ All {n_total:,} barcodes are unique")
    else:
        print(f"  ⚠ Warning: Only {n_unique:,} unique out of {n_total:,} total")
    
    print(f"  ✓ Unique bar_1 used: {len(stats['bar1_usage'])}")
    print(f"  ✓ Unique bar_2 used: {len(stats['bar2_usage'])}")
    
    # Export
    print(f"\n[6/6] Exporting results...")
    output_defaults = _build_default_output_paths(config_file, rows, cols)
    config_output = config.get('output', {})

    if output_prefix:
        # Prefix is interpreted relative to current working directory unless absolute.
        prefix = output_prefix if os.path.isabs(output_prefix) else os.path.normpath(output_prefix)
        layout_file = f"{prefix}.csv"
        barcode_file = f"{prefix}.txt"
        stats_file = f"{prefix}.json"
    elif rows_override is not None or cols_override is not None:
        # If dimensions are overridden, use dimension-specific defaults to avoid
        # clobbering the original config output files.
        layout_file = output_defaults['layout_file']
        barcode_file = output_defaults['barcode_file']
        stats_file = output_defaults['stats_file']
    else:
        layout_file = _resolve_path(config_file, config_output.get('layout_file', output_defaults['layout_file']))
        barcode_file = _resolve_path(config_file, config_output.get('barcode_file', output_defaults['barcode_file']))
        stats_file = _resolve_path(config_file, config_output.get('stats_file', output_defaults['stats_file']))
    
    # Export layout CSV with full details (column-major)
    import csv
    for out_path in (layout_file, barcode_file, stats_file):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(layout_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'col', 'barcode', 'bar_1', 'bar_2'])
        for col in range(cols):
            for row in range(rows):
                combined = grid_combined[row, col]
                b1 = grid_bar1[row, col]
                b2 = grid_bar2[row, col]
                writer.writerow([row, col, combined, b1, b2])
    print(f"  ✓ Layout exported: {layout_file}")
    
    # Export barcode list (column-major order)
    with open(barcode_file, 'w') as f:
        for barcode in barcodes_list:
            f.write(f"{barcode}\n")
    print(f"  ✓ Barcodes exported: {barcode_file}")
    
    # Export statistics
    stats_output = {
        'generation_timestamp': datetime.now().isoformat(),
        'source_config': os.path.abspath(config_file),
        'configuration': config,
        'effective_parameters': {
            'rows': rows,
            'cols': cols,
            'block_rows': block_rows,
            'block_cols': block_cols,
            'seed': seed,
        },
        'results': {
            'total_spots': rows * cols,
            'total_barcodes': len(barcodes_list),
            'unique_barcodes': len(set(barcodes_list)),
            'barcode_length': len(barcodes_list[0]),
            'n_blocks': stats['n_blocks'],
            'unique_bar_1': len(stats['bar1_usage']),
            'unique_bar_2': len(stats['bar2_usage']),
            'all_unique': len(set(barcodes_list)) == len(barcodes_list),
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"  ✓ Statistics saved: {stats_file}")
    
    print(f"\n{'='*80}")
    print("DUAL LAYOUT GENERATION COMPLETE")
    print(f"{'='*80}\n")

    return barcodes_list, stats_output


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Dual barcode layouts with configurable dimensions."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file (barcodes + output settings).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Override array rows from config.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Override array cols from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for deterministic layout generation (default: 1).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output prefix (relative to config dir unless absolute). "
            "When set, outputs are written as <prefix>.csv/.txt/.json"
        ),
    )
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    generate_layout(
        config_file=args.config,
        rows_override=args.rows,
        cols_override=args.cols,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )


if __name__ == '__main__':
    main()
