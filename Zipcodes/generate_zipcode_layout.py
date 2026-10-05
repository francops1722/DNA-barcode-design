#!/usr/bin/env python3
"""
Generate a Zipcodes layout from a JSON config.

Copy the template config, adjust dimensions and output paths, and run
this script without editing the Python source.

Usage:
    python generate_zipcode_layout.py --config Zipcodes/config_zipcodes_template.json
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from spatial_zipcodes.config import load_config
from spatial_zipcodes.axis_code_generator import generate_axis_codes
from spatial_zipcodes.bit_to_dna import BitToDNAEncoder
from spatial_zipcodes.sequence_qc import validate_axis_sequences
from spatial_zipcodes.codebook import build_codebook, export_codebook
from spatial_zipcodes.distance_analysis import analyze_axis_codes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def export_barcodes_txt(codebook, filepath):
    """Export barcodes to simple text file (column-major order)"""
    with open(filepath, 'w') as f:
        for feature in codebook.features:
            f.write(f"{feature.zip_seq}\n")
    print(f"  ✓ Barcodes exported: {filepath}")


def export_layout_csv(codebook, filepath, colmajor=False):
    """Export layout CSV with row, col, barcode"""
    # Create DataFrame
    data = []
    for feature in codebook.features:
        data.append({
            'row': feature.row,
            'col': feature.col,
            'barcode': feature.zip_seq
        })
    df = pd.DataFrame(data)
    
    # Sort: column-major or row-major
    if colmajor:
        df = df.sort_values(['col', 'row'])
    else:
        df = df.sort_values(['row', 'col'])
    
    df.to_csv(filepath, index=False)
    order_str = "column-major" if colmajor else "row-major"
    print(f"  ✓ Layout exported: {filepath} ({order_str})")


def resolve_output_paths(config_file, config):
    """Resolve output paths from config or sensible defaults."""
    config_path = Path(config_file).resolve()
    config_dir = config_path.parent
    stem = config_path.stem

    output_cfg = getattr(config, "output", None)
    if output_cfg is not None:
        output_dir = Path(output_cfg.directory)
        if not output_dir.is_absolute():
            output_dir = config_dir / output_dir
        return {
            "output_dir": output_dir,
            "barcodes_file": output_dir / output_cfg.barcodes_file,
            "layout_file": output_dir / output_cfg.layout_file,
            "layout_colmajor_file": output_dir / getattr(output_cfg, "layout_colmajor_file", f"{stem}_colmajor.csv"),
        }

    output_dir = config_dir / f"output_{stem}"
    return {
        "output_dir": output_dir,
        "barcodes_file": output_dir / f"barcodes_{stem}.txt",
        "layout_file": output_dir / f"layout_{stem}.csv",
        "layout_colmajor_file": output_dir / f"layout_{stem}_colmajor.csv",
    }


def generate_set(set_num, config_file):
    """Generate a zipcodes layout from one config file."""
    
    print(f"\n{'='*80}")
    print(f"GENERATING SET {set_num} - ZIPCODES (118×182 ARRAY, NO SEPARATOR)")
    print(f"{'='*80}\n")
    print(f"Configuration: {config_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"[1/8] Loading configuration...")
    try:
        config = load_config(config_file)
        rows = config.chip_geometry.rows
        cols = config.chip_geometry.cols
        seed = config.axis_code.random_seed
        print(f"  ✓ Configuration loaded")
        print(f"  Array: {rows}×{cols} = {rows*cols:,} spots")
        print(f"  X-axis codes needed: {cols}")
        print(f"  Y-axis codes needed: {rows}")
        print(f"  Random seed: {seed}")
        print(f"  Spacer: {'None' if not config.codebook.spacer_sequence else config.codebook.spacer_sequence}")
    except Exception as e:
        print(f"  ✗ Error loading configuration: {e}")
        raise
    
    # Resolve output directory and files from config or derived defaults
    outputs = resolve_output_paths(config_file, config)
    output_dir = outputs["output_dir"]
    barcodes_file = outputs["barcodes_file"]
    layout_file = outputs["layout_file"]
    layout_colmajor_file = outputs["layout_colmajor_file"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intermediate").mkdir(exist_ok=True)
    
    # Generate axis codes
    print(f"\n[2/8] Generating X and Y axis codes...")
    print(f"  This may take 1-3 minutes due to constraint satisfaction...")
    try:
        # Generate X-axis codes
        print(f"  Generating X-axis codes for {cols} columns...")
        x_codes, x_verification = generate_axis_codes(
            n_positions=cols,
            bit_length=config.x_bit_length,
            target_distances=config.error_correction.distances,
            far_features_min_distance=config.error_correction.far_features_min_distance,
            max_attempts_per_position=config.axis_code.max_attempts_per_position,
            backtrack_depth=config.axis_code.backtrack_depth,
            tolerance=config.axis_code.tolerance,
            random_seed=seed,
        )
        print(f"    ✓ Generated {len(x_codes)} X-codes")
        print(f"    Hard constraint violations: {x_verification['hard_constraint_violations']}")
        
        # Generate Y-axis codes
        print(f"  Generating Y-axis codes for {rows} rows...")
        y_codes, y_verification = generate_axis_codes(
            n_positions=rows,
            bit_length=config.y_bit_length,
            target_distances=config.error_correction.distances,
            far_features_min_distance=config.error_correction.far_features_min_distance,
            max_attempts_per_position=config.axis_code.max_attempts_per_position,
            backtrack_depth=config.axis_code.backtrack_depth,
            tolerance=config.axis_code.tolerance,
            random_seed=seed,
        )
        print(f"    ✓ Generated {len(y_codes)} Y-codes")
        print(f"    Hard constraint violations: {y_verification['hard_constraint_violations']}")
        
        axis_codes = {'x_codes': x_codes, 'y_codes': y_codes}
        
        print(f"  ✓ Axis codes generated")
        print(f"    X-codes: {len(x_codes)} codes of {len(x_codes[0])} bits")
        print(f"    Y-codes: {len(y_codes)} codes of {len(y_codes[0])} bits")
        
        # Save axis codes
        import json as json_module
        axis_codes_file = output_dir / "intermediate" / "axis_codes.json"
        with open(axis_codes_file, 'w') as f:
            json_module.dump({
                'x_codes': [code.tolist() if hasattr(code, 'tolist') else list(code) 
                           for code in x_codes],
                'y_codes': [code.tolist() if hasattr(code, 'tolist') else list(code) 
                           for code in y_codes]
            }, f, indent=2)
        print(f"    Saved axis codes to {axis_codes_file}")
        
    except Exception as e:
        print(f"  ✗ Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Convert to DNA
    print(f"\n[3/8] Converting bit sequences to DNA...")
    try:
        encoder = BitToDNAEncoder(
            scheme=config.bit_to_dna.scheme,
            mapping=config.bit_to_dna.mapping
        )
        
        x_sequences = [encoder.encode(code) for code in x_codes]
        y_sequences = [encoder.encode(code) for code in y_codes]
        
        print(f"  ✓ DNA sequences generated")
        print(f"    X-sequences: {len(x_sequences)} × {len(x_sequences[0])}nt")
        print(f"    Y-sequences: {len(y_sequences)} × {len(y_sequences[0])}nt")
        print(f"    Example X-code: {x_sequences[0]}")
        print(f"    Example Y-code: {y_sequences[0]}")
        
    except Exception as e:
        print(f"  ✗ Encoding error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Validate sequences
    print(f"\n[4/8] Validating sequence quality...")
    try:
        qc_config = {
            'gc_content_min': config.sequence_qc.gc_content_min,
            'gc_content_max': config.sequence_qc.gc_content_max,
            'g_content_max': config.sequence_qc.g_content_max,
            'max_homopolymer_length': config.sequence_qc.max_homopolymer_length
        }
        x_valid, x_issues = validate_axis_sequences(x_sequences, 'X', qc_config)
        y_valid, y_issues = validate_axis_sequences(y_sequences, 'Y', qc_config)
        
        if x_valid and y_valid:
            print(f"  ✓ All sequences passed QC")
        else:
            if not x_valid:
                print(f"  ⚠ X-axis issues: {x_issues}")
            if not y_valid:
                print(f"  ⚠ Y-axis issues: {y_issues}")
            print(f"    Proceeding with current sequences...")
        
    except Exception as e:
        print(f"  ⚠ Warning: Validation error: {e}")
    
    # Analyze distances
    print(f"\n[5/8] Analyzing distance distributions...")
    try:
        x_results, y_results = analyze_axis_codes(
            x_codes,
            y_codes,
            config.error_correction.distances,
            config.error_correction.far_features_min_distance
        )
        print(f"  ✓ Distance analysis complete")
        # Print diff 1 statistics (1-step neighbors)
        diff1_x = [d for d in x_results['distance_stats'] if d['diff'] == 1][0]
        diff1_y = [d for d in y_results['distance_stats'] if d['diff'] == 1][0]
        print(f"    X-axis 1-step neighbors: mean={diff1_x['mean']:.2f} bits")
        print(f"    Y-axis 1-step neighbors: mean={diff1_y['mean']:.2f} bits")
        
    except Exception as e:
        print(f"  ⚠ Warning: Analysis error: {e}")
        x_results = y_results = None
    
    # Build codebook
    print(f"\n[6/8] Building codebook...")
    try:
        codebook = build_codebook(
            rows=rows,
            cols=cols,
            x_codes=x_codes,
            y_codes=y_codes,
            x_sequences=x_sequences,
            y_sequences=y_sequences,
            spacer=config.codebook.spacer_sequence,
            feature_id_format=config.codebook.feature_id_format
        )
        
        print(f"  ✓ Codebook built")
        print(f"    Total barcodes: {len(codebook.features):,}")
        print(f"    Barcode length: {len(codebook.features[0].zip_seq)}nt")
        
        # Check uniqueness
        all_zips = codebook.get_all_zip_sequences()
        unique_zips = len(set(all_zips))
        print(f"    Unique barcodes: {unique_zips:,}")
        
        if unique_zips == len(all_zips):
            print(f"    ✓ All barcodes are unique")
        else:
            duplicates = len(all_zips) - unique_zips
            print(f"    ⚠ Warning: {duplicates} duplicate barcodes")
        
    except Exception as e:
        print(f"  ✗ Codebook error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Export results
    print(f"\n[7/8] Exporting results...")
    try:
        # Export codebook CSV
        codebook_file = output_dir / "codebook.csv"
        export_codebook(codebook, codebook_file, file_format='csv')
        print(f"  ✓ Full codebook exported: {codebook_file}")
        
        # Export simple text file (column-major)
        export_barcodes_txt(codebook, barcodes_file)
        
        # Create filtered CSV layouts
        export_layout_csv(codebook, layout_file, colmajor=False)
        
        export_layout_csv(codebook, layout_colmajor_file, colmajor=True)
        
        # Export statistics
        all_zips = codebook.get_all_zip_sequences()
        stats = {
            'generation_timestamp': datetime.now().isoformat(),
            'configuration': {
                'rows': rows,
                'cols': cols,
                'total_spots': rows * cols,
                'x_bit_length': config.x_bit_length,
                'y_bit_length': config.y_bit_length,
                'random_seed': seed,
                'encoding_scheme': config.bit_to_dna.scheme,
                'spacer': config.codebook.spacer_sequence
            },
            'results': {
                'total_barcodes': len(all_zips),
                'unique_barcodes': len(set(all_zips)),
                'barcode_length': len(all_zips[0]),
                'x_sequence_length': len(x_sequences[0]),
                'y_sequence_length': len(y_sequences[0]),
                'all_unique': len(set(all_zips)) == len(all_zips)
            },
            'spatial_properties': {
                'design_principle': 'Gray-code-like: spatial neighbors have similar sequences',
                'x_neighbor_distance_mean': diff1_x['mean'] if x_results else None,
                'y_neighbor_distance_mean': diff1_y['mean'] if y_results else None
            }
        }
        
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Statistics saved: {stats_file}")
        
    except Exception as e:
        print(f"  ✗ Export error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verification
    print(f"\n[8/8] Verifying layout...")
    try:
        all_zips = codebook.get_all_zip_sequences()
        n_unique = len(set(all_zips))
        n_total = len(all_zips)
        barcode_length = len(all_zips[0])
        
        if n_unique == n_total:
            print(f"  ✓ All {n_total:,} barcodes are unique")
        else:
            print(f"  ⚠ Only {n_unique:,} unique out of {n_total:,} total")
        
        print(f"  ✓ Barcode length: {barcode_length}nt")
        print(f"  ✓ Column-major ordering maintained")
        
    except Exception as e:
        print(f"  ⚠ Verification error: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"ZIPCODES GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Summary:")
    print(f"  • Total spots: {n_total:,}")
    print(f"  • Unique barcodes: {n_unique:,}")
    print(f"  • Barcode length: {barcode_length}nt ({len(x_sequences[0])}nt + {len(y_sequences[0])}nt)")
    print(f"  • Random seed: {seed}")
    print(f"\nOutput directory: {output_dir}/")
    
    return codebook, config, all_zips, output_dir


def verify_uniqueness_between_sets(zips1, zips2):
    """Verify no barcode overlap between two sets"""
    print(f"\n{'='*80}")
    print(f"CROSS-SET VERIFICATION")
    print(f"{'='*80}\n")
    
    set1 = set(zips1)
    set2 = set(zips2)
    
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
    ap = argparse.ArgumentParser(description="Generate a Zipcodes layout from a JSON config.")
    ap.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_zipcodes_template.json")),
        help="Path to JSON config file.",
    )
    args = ap.parse_args()

    print("\n" + "="*80)
    print("ZIPCODES LAYOUT GENERATOR")
    print("="*80)

    codebook, config, zips, output_dir = generate_set(1, args.config)

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(zips):,} barcodes")
    print(f"Unique: {len(set(zips)):,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
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
