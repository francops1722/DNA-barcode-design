"""
Command-line interface for spatial zipcode tools.

Provides commands for generating codes, analyzing distances,
building codebooks, and simulating errors.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

from .config import load_config, save_config
from .axis_code_generator import generate_axis_codes
from .bit_to_dna import BitToDNAEncoder, encode_codes_to_dna
from .sequence_qc import validate_axis_sequences
from .codebook import build_codebook, export_codebook
from .distance_analysis import analyze_axis_codes, AxisDistanceAnalyzer
from .simulation import (
    SubstitutionErrorModel,
    IndelErrorModel,
    ZipcodeDecoder,
    ZipcodeSimulator,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def cmd_generate_axis_codes(args):
    """Generate axis codes from configuration."""
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate X codes
    logger.info("Generating X axis codes...")
    x_codes, x_verification = generate_axis_codes(
        n_positions=config.chip_geometry.cols,
        bit_length=config.x_bit_length,
        target_distances=config.error_correction.distances,
        far_features_min_distance=config.error_correction.far_features_min_distance,
        max_attempts_per_position=config.axis_code.max_attempts_per_position,
        backtrack_depth=config.axis_code.backtrack_depth,
        tolerance=config.axis_code.tolerance,
        random_seed=config.axis_code.random_seed,
    )
    
    # Generate Y codes
    logger.info("Generating Y axis codes...")
    y_codes, y_verification = generate_axis_codes(
        n_positions=config.chip_geometry.rows,
        bit_length=config.y_bit_length,
        target_distances=config.error_correction.distances,
        far_features_min_distance=config.error_correction.far_features_min_distance,
        max_attempts_per_position=config.axis_code.max_attempts_per_position,
        backtrack_depth=config.axis_code.backtrack_depth,
        tolerance=config.axis_code.tolerance,
        random_seed=config.axis_code.random_seed,
    )
    
    # Save codes
    logger.info("Saving axis codes...")
    
    # Save as JSON
    codes_data = {
        'x_codes': [[int(b) for b in code] for code in x_codes],
        'y_codes': [[int(b) for b in code] for code in y_codes],
        'x_verification': {
            k: v for k, v in x_verification.items()
            if k not in ['distance_distribution', 'soft_constraint_stats']
        },
        'y_verification': {
            k: v for k, v in y_verification.items()
            if k not in ['distance_distribution', 'soft_constraint_stats']
        },
    }
    
    with open(output_dir / 'axis_codes.json', 'w') as f:
        json.dump(codes_data, f, indent=2)
    
    logger.info(f"Axis codes saved to {output_dir / 'axis_codes.json'}")
    
    # Print summary
    print(f"\nX axis: {len(x_codes)} codes, {len(x_codes[0])} bits each")
    print(f"  Hard constraint violations: {x_verification['hard_constraint_violations']}")
    
    print(f"\nY axis: {len(y_codes)} codes, {len(y_codes[0])} bits each")
    print(f"  Hard constraint violations: {y_verification['hard_constraint_violations']}")


def cmd_analyze_distances(args):
    """Analyze distance distributions from axis codes."""
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    logger.info(f"Loading axis codes from {args.codes}")
    with open(args.codes, 'r') as f:
        codes_data = json.load(f)
    
    x_codes = codes_data['x_codes']
    y_codes = codes_data['y_codes']
    
    # Analyze
    x_results, y_results = analyze_axis_codes(
        x_codes=x_codes,
        y_codes=y_codes,
        target_distances=config.error_correction.distances,
        far_features_min_distance=config.error_correction.far_features_min_distance,
    )
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {
            'x_analysis': x_results,
            'y_analysis': y_results,
        }
        
        with open(output_dir / 'distance_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_dir / 'distance_analysis.json'}")


def cmd_build_codebook(args):
    """Build complete codebook from axis codes."""
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    logger.info(f"Loading axis codes from {args.codes}")
    with open(args.codes, 'r') as f:
        codes_data = json.load(f)
    
    x_codes = codes_data['x_codes']
    y_codes = codes_data['y_codes']
    
    # Encode to DNA
    logger.info("Encoding axis codes to DNA sequences...")
    encoder = BitToDNAEncoder(
        scheme=config.bit_to_dna.scheme,
        mapping=config.bit_to_dna.mapping,
    )
    
    x_sequences = [encoder.encode(code) for code in x_codes]
    y_sequences = [encoder.encode(code) for code in y_codes]
    
    # QC checks
    if not args.skip_qc:
        logger.info("Performing sequence QC...")
        
        from dataclasses import asdict
        qc_config = asdict(config.sequence_qc)
        
        x_passed, x_qc_results = validate_axis_sequences(
            x_sequences, "X", qc_config
        )
        y_passed, y_qc_results = validate_axis_sequences(
            y_sequences, "Y", qc_config
        )
        
        if not x_passed or not y_passed:
            logger.warning("Some sequences failed QC checks")
            if args.strict:
                logger.error("Strict mode: aborting due to QC failures")
                sys.exit(1)
    
    # Build codebook
    logger.info("Building codebook...")
    codebook = build_codebook(
        rows=config.chip_geometry.rows,
        cols=config.chip_geometry.cols,
        x_codes=x_codes,
        y_codes=y_codes,
        x_sequences=x_sequences,
        y_sequences=y_sequences,
        spacer=config.codebook.spacer_sequence,
        feature_id_format=config.codebook.feature_id_format,
    )
    
    # Export codebook
    output_path = Path(args.output)
    logger.info(f"Exporting codebook to {output_path}...")
    
    export_codebook(
        codebook=codebook,
        output_path=output_path,
        file_format=config.codebook.output_format,
    )
    
    # Print statistics
    stats = codebook.get_statistics()
    print(f"\nCodebook Statistics:")
    print(f"  Features: {stats['n_features']}")
    print(f"  Grid: {stats['rows']} x {stats['cols']}")
    print(f"  X: {stats['x_bit_length']} bits -> {stats['x_dna_length']} bp")
    print(f"  Y: {stats['y_bit_length']} bits -> {stats['y_dna_length']} bp")
    print(f"  Spacer: '{stats['spacer']}' ({stats['spacer_length']} bp)")
    print(f"  Full zip code: {stats['zip_length']} bp")
    
    logger.info("Codebook generation complete")


def cmd_simulate_errors(args):
    """Simulate sequencing errors and test decoding."""
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    logger.info(f"Loading codebook from {args.codebook}")
    
    # Load codebook
    from .codebook import load_codebook_from_csv, Codebook
    
    delimiter = '\t' if str(args.codebook).endswith('.tsv') else ','
    features = load_codebook_from_csv(args.codebook, delimiter=delimiter)
    
    # Reconstruct codebook object (simplified)
    # Extract unique X and Y sequences
    x_sequences = []
    y_sequences = []
    x_codes = []
    y_codes = []
    
    seen_cols = set()
    seen_rows = set()
    
    for feature in features:
        if feature.col not in seen_cols:
            x_sequences.append(feature.x_seq)
            x_codes.append([int(b) for b in feature.x_bits])
            seen_cols.add(feature.col)
        if feature.row not in seen_rows:
            y_sequences.append(feature.y_seq)
            y_codes.append([int(b) for b in feature.y_bits])
            seen_rows.add(feature.row)
    
    codebook = Codebook(
        rows=config.chip_geometry.rows,
        cols=config.chip_geometry.cols,
        x_codes=x_codes,
        y_codes=y_codes,
        x_sequences=x_sequences,
        y_sequences=y_sequences,
        spacer=config.codebook.spacer_sequence,
        feature_id_format=config.codebook.feature_id_format,
    )
    
    # Create error model
    logger.info("Setting up error model...")
    if config.simulation.error_model == "substitution":
        error_model = SubstitutionErrorModel(
            error_rate=config.simulation.per_base_error_rate,
            random_seed=config.axis_code.random_seed,
        )
    elif config.simulation.error_model == "indel":
        error_model = IndelErrorModel(
            substitution_rate=config.simulation.per_base_error_rate,
            insertion_rate=config.simulation.per_base_error_rate / 2,
            deletion_rate=config.simulation.per_base_error_rate / 2,
            random_seed=config.axis_code.random_seed,
        )
    else:
        logger.error(f"Unknown error model: {config.simulation.error_model}")
        sys.exit(1)
    
    # Create decoder
    decoder = ZipcodeDecoder(
        codebook=codebook,
        distance_metric=config.simulation.distance_metric,
        max_edit_distance=config.simulation.max_edit_distance_for_decoding,
        spacer=config.codebook.spacer_sequence,
    )
    
    # Create simulator
    simulator = ZipcodeSimulator(
        codebook=codebook,
        error_model=error_model,
        decoder=decoder,
        random_seed=config.axis_code.random_seed,
    )
    
    # Run simulation
    logger.info(f"Running simulation (n_samples={args.n_samples})...")
    results = simulator.simulate_batch(
        n_samples=args.n_samples,
        sample_all=args.all,
    )
    
    # Print summary
    simulator.print_summary(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove individual results for smaller file
        results_to_save = {k: v for k, v in results.items() if k != 'individual_results'}
        results_to_save['n_individual_results'] = len(results['individual_results'])
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_to_save = convert_numpy(results_to_save)
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Simulation results saved to {output_path}")


def cmd_full_pipeline(args):
    """Run complete pipeline: generate codes -> build codebook."""
    logger.info("Running full pipeline")
    
    # Generate axis codes
    logger.info("Step 1: Generating axis codes")
    args_gen = argparse.Namespace(
        config=args.config,
        output=args.output_dir / "intermediate",
    )
    cmd_generate_axis_codes(args_gen)
    
    # Analyze distances
    logger.info("Step 2: Analyzing distances")
    args_analyze = argparse.Namespace(
        config=args.config,
        codes=args.output_dir / "intermediate" / "axis_codes.json",
        output=args.output_dir / "intermediate",
    )
    cmd_analyze_distances(args_analyze)
    
    # Build codebook
    logger.info("Step 3: Building codebook")
    args_codebook = argparse.Namespace(
        config=args.config,
        codes=args.output_dir / "intermediate" / "axis_codes.json",
        output=args.output_dir / "codebook.csv",
        skip_qc=args.skip_qc,
        strict=args.strict,
    )
    cmd_build_codebook(args_codebook)
    
    # Optionally run simulation
    if not args.skip_simulation:
        logger.info("Step 4: Running error simulation")
        args_sim = argparse.Namespace(
            config=args.config,
            codebook=args.output_dir / "codebook.csv",
            n_samples=1000,
            all=False,
            output=args.output_dir / "simulation_results.json",
        )
        cmd_simulate_errors(args_sim)
    
    logger.info(f"Pipeline complete. Outputs in {args.output_dir}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Spatial Zipcodes - Tools for generating error-correcting spatial barcodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate axis codes
    parser_gen = subparsers.add_parser(
        'generate-axis-codes',
        help='Generate X and Y axis codes',
    )
    parser_gen.add_argument('--config', required=True, help='Configuration JSON file')
    parser_gen.add_argument('--output', required=True, help='Output directory')
    parser_gen.set_defaults(func=cmd_generate_axis_codes)
    
    # Analyze distances
    parser_analyze = subparsers.add_parser(
        'analyze-distances',
        help='Analyze distance distributions',
    )
    parser_analyze.add_argument('--config', required=True, help='Configuration JSON file')
    parser_analyze.add_argument('--codes', required=True, help='Axis codes JSON file')
    parser_analyze.add_argument('--output', help='Output directory for results')
    parser_analyze.set_defaults(func=cmd_analyze_distances)
    
    # Build codebook
    parser_codebook = subparsers.add_parser(
        'build-codebook',
        help='Build complete codebook',
    )
    parser_codebook.add_argument('--config', required=True, help='Configuration JSON file')
    parser_codebook.add_argument('--codes', required=True, help='Axis codes JSON file')
    parser_codebook.add_argument('--output', required=True, help='Output codebook file')
    parser_codebook.add_argument('--skip-qc', action='store_true', help='Skip QC checks')
    parser_codebook.add_argument('--strict', action='store_true', help='Abort on QC failures')
    parser_codebook.set_defaults(func=cmd_build_codebook)
    
    # Simulate errors
    parser_sim = subparsers.add_parser(
        'simulate-errors',
        help='Simulate sequencing errors and test decoding',
    )
    parser_sim.add_argument('--config', required=True, help='Configuration JSON file')
    parser_sim.add_argument('--codebook', required=True, help='Codebook CSV/TSV file')
    parser_sim.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser_sim.add_argument('--all', action='store_true', help='Simulate all features')
    parser_sim.add_argument('--output', help='Output JSON file for results')
    parser_sim.set_defaults(func=cmd_simulate_errors)
    
    # Full pipeline
    parser_full = subparsers.add_parser(
        'full-pipeline',
        help='Run complete pipeline',
    )
    parser_full.add_argument('--config', required=True, help='Configuration JSON file')
    parser_full.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser_full.add_argument('--skip-qc', action='store_true', help='Skip QC checks')
    parser_full.add_argument('--strict', action='store_true', help='Abort on QC failures')
    parser_full.add_argument('--skip-simulation', action='store_true', help='Skip simulation step')
    parser_full.set_defaults(func=cmd_full_pipeline)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
