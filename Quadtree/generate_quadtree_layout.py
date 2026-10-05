#!/usr/bin/env python3
"""
Generate a Quadtree+Library layout from a JSON config.

Copy the template config, adjust dimensions and output paths, then run
this script with --config.

Usage:
    python generate_quadtree_layout.py --config Quadtree/config_quadtree_template.json
"""

import argparse
import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_hybrid_quadtree_library import generate_hybrid_layout, analyze_hybrid_layout


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def export_barcodes_txt(df, filepath):
    """Export barcodes to simple text file (column-major order)"""
    with open(filepath, 'w') as f:
        for col in sorted(df['col'].unique()):
            col_data = df[df['col'] == col].sort_values('row')
            for barcode in col_data['barcode']:
                f.write(f"{barcode}\n")
    
    print(f"  ✓ Barcodes exported: {filepath}")


def save_statistics(all_stats, config, df, filepath):
    """Save comprehensive statistics to JSON"""
    
    # Compute Hamming distance statistics
    hamming_dists = []
    rows = df['row'].max() + 1
    cols = df['col'].max() + 1
    
    sample_size = min(1000, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    for idx in sample_indices:
        row, col = df.iloc[idx]['row'], df.iloc[idx]['col']
        quadtree_bc = df.iloc[idx]['quadtree_combined']
        
        if col < cols - 1:
            neighbor = df[(df['row'] == row) & (df['col'] == col + 1)]
            if len(neighbor) > 0:
                neighbor_bc = neighbor['quadtree_combined'].values[0]
                dist = sum(c1 != c2 for c1, c2 in zip(quadtree_bc, neighbor_bc))
                hamming_dists.append(dist)
        
        if row < rows - 1:
            neighbor = df[(df['row'] == row + 1) & (df['col'] == col)]
            if len(neighbor) > 0:
                neighbor_bc = neighbor['quadtree_combined'].values[0]
                dist = sum(c1 != c2 for c1, c2 in zip(quadtree_bc, neighbor_bc))
                hamming_dists.append(dist)
    
    output = {
        'generation_timestamp': datetime.now().isoformat(),
        'configuration': config,
        'layout_statistics': {
            'total_spots': len(df),
            'unique_combined_barcodes': df['barcode'].nunique(),
            'total_barcode_length': len(df['barcode'].iloc[0]),
            'quadtree_length': len(df['quadtree_combined'].iloc[0]),
            'library_length': len(df['library'].iloc[0]),
            'grid_dimensions': {
                'rows': int(df['row'].max() + 1),
                'cols': int(df['col'].max() + 1)
            }
        },
        'quadtree_statistics': {
            'unique_quadtree_codes': df['quadtree_combined'].nunique(),
            'rll_stats': all_stats
        },
        'library_statistics': {
            'unique_library_codes': df['library'].nunique(),
            'average_reuse': len(df) / df['library'].nunique()
        },
        'spatial_similarity': {
            'neighbor_hamming_distances_quadtree_only': {
                'mean': float(np.mean(hamming_dists)) if hamming_dists else 0,
                'std': float(np.std(hamming_dists)) if hamming_dists else 0,
                'min': int(np.min(hamming_dists)) if hamming_dists else 0,
                'max': int(np.max(hamming_dists)) if hamming_dists else 0,
                'sample_size': len(hamming_dists)
            }
        },
        'quality_metrics': {
            'all_unique': df['barcode'].nunique() == len(df),
            'no_duplicates': not df.duplicated(subset=['barcode']).any()
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  ✓ Statistics saved: {filepath}")


def generate_set(set_num, config_file):
    """Generate hybrid layout for one set"""
    
    print(f"\n{'='*80}")
    print(f"GENERATING SET {set_num} - HYBRID QUADTREE+LIBRARY (118×182 ARRAY)")
    print(f"{'='*80}\n")
    print(f"Configuration: {config_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"[1/6] Loading configuration...")
    config = load_config(config_file)
    
    rows = config['array']['rows']
    cols = config['array']['cols']
    total_spots = rows * cols
    
    quadtree_length = config['quadtree']['quadtree_length_nt']
    library_file = config['library']['file']
    library_length = config['library']['length_nt']
    total_length = config['barcode']['total_length_nt']
    max_run = config['rll']['max_run']
    seed = config['random_seed']
    
    output_dir = Path(config['output']['directory'])
    output_csv = output_dir / config['output']['layout_file']
    output_txt = output_dir / config['output']['barcodes_file']
    output_stats = output_dir / config['output']['stats_file']
    
    print(f"  Array: {rows}×{cols} = {total_spots:,} spots")
    print(f"  Barcode: {quadtree_length}nt quadtree + {library_length}nt library = {total_length}nt")
    print(f"  Library: {library_file}")
    print(f"  RLL max_run: {max_run}")
    print(f"  Random seed: {seed}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate layout
    print(f"\n[2/6] Generating hybrid layout...")
    try:
        df, quadtree_segments, all_stats = generate_hybrid_layout(
            rows, cols,
            library_file=str(library_file),
            segment_length=8,
            n_quadtree_rounds=3,
            assignment_strategy='random',
            max_run=max_run
        )
        print(f"  ✓ Layout generated: {len(df):,} barcodes")
        
    except Exception as e:
        print(f"  ✗ Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Analyze layout
    print(f"\n[3/6] Analyzing layout quality...")
    try:
        analyze_hybrid_layout(df, n_quadtree_rounds=3, all_stats=all_stats)
        print(f"  ✓ Analysis complete")
        
    except Exception as e:
        print(f"  ⚠ Warning: Analysis error: {e}")
    
    # Export files
    print(f"\n[4/6] Exporting results...")
    try:
        # Export CSV (column-major order)
        df_export = df.copy()
        df_export = df_export.sort_values(['col', 'row'])
        df_export.to_csv(output_csv, index=False)
        print(f"  ✓ Layout CSV exported: {output_csv}")
        
        file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
        print(f"    File size: {file_size_mb:.2f} MB")
        
        # Export simple text file
        export_barcodes_txt(df_export, output_txt)
        
        # Export statistics
        save_statistics(all_stats, config, df, output_stats)
        
    except Exception as e:
        print(f"  ✗ Export error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verification
    print(f"\n[5/6] Verifying layout...")
    n_unique = df['barcode'].nunique()
    n_total = len(df)
    library_unique = df['library'].nunique()
    quadtree_unique = df['quadtree_combined'].nunique()
    
    if n_unique == n_total:
        print(f"  ✓ All {n_total:,} combined barcodes are unique")
    else:
        print(f"  ⚠ Warning: Only {n_unique:,} unique out of {n_total:,} total")
    
    barcode_length = len(df['barcode'].iloc[0])
    if barcode_length == total_length:
        print(f"  ✓ All barcodes are {barcode_length}nt")
    else:
        print(f"  ⚠ Warning: Expected {total_length}nt, got {barcode_length}nt")
    
    print(f"  ✓ Using {quadtree_unique:,} unique quadtree codes")
    print(f"  ✓ Using {library_unique:,} unique library codes")
    
    # Summary
    print(f"\n[6/6] Summary for Set {set_num}:")
    print(f"  • Total spots: {n_total:,}")
    print(f"  • Unique barcodes: {n_unique:,}")
    print(f"  • Barcode length: {barcode_length}nt ({quadtree_length}nt + {library_length}nt)")
    print(f"  • Quadtree codes: {quadtree_unique:,} unique")
    print(f"  • Library codes: {library_unique:,} unique")
    print(f"  • Random seed: {seed}")
    
    overall_stats = all_stats[-1] if all_stats else None
    if overall_stats:
        print(f"  • RLL forced flips: {overall_stats['total_forced_flips']}")
        print(f"  • RLL violations avoided: {overall_stats['total_violations_avoided']}")
    
    print(f"\n{'='*80}")
    print(f"SET {set_num} GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    return df, config


def verify_uniqueness_between_sets(df1, df2):
    """Verify no barcode overlap between two sets"""
    print(f"\n{'='*80}")
    print(f"CROSS-SET VERIFICATION")
    print(f"{'='*80}\n")
    
    set1 = set(df1['barcode'])
    set2 = set(df2['barcode'])
    
    overlap = set1 & set2
    
    print(f"Set 1: {len(set1):,} unique barcodes")
    print(f"Set 2: {len(set2):,} unique barcodes")
    print(f"Overlap: {len(overlap):,} barcodes")
    
    if len(overlap) == 0:
        print(f"\n✓ NO OVERLAP - Both sets are completely unique!")
        return True
    else:
        print(f"\n✗ OVERLAP DETECTED - {len(overlap)} barcodes appear in both sets:")
        for i, bc in enumerate(list(overlap)[:10]):
            print(f"  {i+1}. {bc}")
        if len(overlap) > 10:
            print(f"  ... and {len(overlap)-10} more")
        return False


def main():
    """Main execution."""
    ap = argparse.ArgumentParser(description="Generate a Quadtree+Library layout from a JSON config.")
    ap.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_quadtree_template.json")),
        help="Path to JSON config file.",
    )
    args = ap.parse_args()

    print("\n" + "="*80)
    print("QUADTREE+LIBRARY LAYOUT GENERATOR")
    print("="*80)

    df, config = generate_set(1, args.config)

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(df):,} barcodes")
    print(f"Unique: {len(set(df['barcode'])):,}")
    print(f"Output directory: {config['output']['directory']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
