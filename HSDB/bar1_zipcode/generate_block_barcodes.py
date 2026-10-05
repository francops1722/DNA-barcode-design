#!/usr/bin/env python3
"""
Phase 2 — Generate block barcodes (bar_1) for the HSDB design.

Runs the spatial_zipcodes axis-code generator on the block grid defined in
the config file, then combines X (14nt) + Y (14nt) into a 28nt bar_1
sequence per block.  The same script handles any array size by pointing it
at a different config:

  118×182 array  →  block_zipcode_config.json          (30×46  = 1,380 blocks)
  253×363 array  →  block_zipcode_config_253x363.json  (64×91  = 5,824 blocks)

Spatial gradient property:
  - Adjacent blocks (|Δrow|=1 or |Δcol|=1) differ by >= 4 bases in bar_1.
  - Blocks 5+ steps apart differ by >= 16 bases.

G-content post-filter (Idea 3):
  - Any barcode with G > 25% has excess G positions swapped G→T.
  - Each such swap changes Hamming distance to neighbors by at most 1,
    which is within the tolerance of the spatial gradient.

Outputs (written to --out-dir):
  - bar1_barcodes_28nt.txt   : sequences in row-major block order (one per line)
  - bar1_codebook.csv        : block_row, block_col, bar_1
  - bar1_qc_report.json      : per-barcode QC and distance statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ZIPCODES_DIR = REPO_ROOT / "Zipcodes"

# Make the spatial_zipcodes package importable
if str(ZIPCODES_DIR) not in sys.path:
    sys.path.insert(0, str(ZIPCODES_DIR))

from spatial_zipcodes.config import load_config
from spatial_zipcodes.axis_code_generator import generate_axis_codes, hamming_distance
from spatial_zipcodes.bit_to_dna import BitToDNAEncoder


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Block grid dimensions are read from the config file at runtime.
# These are set as module-level defaults but overridden inside main().
BAR1_LEN = 28                        # 14 nt X + 14 nt Y, no spacer
G_MAX_FRACTION = 0.25                # Idea 3: G <= 25% per barcode


# --------------------------------------------------------------------------
# QC helpers (G content)
# --------------------------------------------------------------------------

def g_fraction(seq: str) -> float:
    return seq.count("G") / len(seq)


def reduce_g_content(seq: str, max_g_frac: float = G_MAX_FRACTION) -> Tuple[str, int]:
    """
    Replace excess G bases with T (minimal perturbation).
    Returns (modified_seq, n_swaps).
    """
    max_g = int(max_g_frac * len(seq))
    g_positions = [i for i, b in enumerate(seq) if b == "G"]
    n_excess = len(g_positions) - max_g
    if n_excess <= 0:
        return seq, 0
    bases = list(seq)
    # Swap the last n_excess G positions to T (arbitrary choice, minimal bias)
    for pos in g_positions[-n_excess:]:
        bases[pos] = "T"
    return "".join(bases), n_excess


def max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    max_run = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def break_homopolymers(
    seq: str,
    max_run: int = 3,
    g_max_frac: float = G_MAX_FRACTION,
) -> Tuple[str, int]:
    """
    Shorten homopolymer runs that exceed *max_run* by substituting a different
    base at every (max_run+1)-th position of the run.

    Substitution priority avoids G when possible (to respect G-content limit)
    and avoids creating a new run of the same length:
      A/T run → break with C  (adds GC, avoids G)
      C/G run → break with A  (adds AT, avoids G)

    Returns (modified_seq, n_substitutions).
    """
    bases = list(seq)
    n_subs = 0
    i = 0
    while i < len(bases):
        # Find end of current run
        j = i
        while j < len(bases) and bases[j] == bases[i]:
            j += 1
        run_len = j - i
        if run_len > max_run:
            run_base = bases[i]
            # Pick a break base
            if run_base in ("A", "T"):
                break_base = "C"
            else:  # C or G
                break_base = "A"
            # Insert break every (max_run) bases within the run
            for k in range(i + max_run, j, max_run + 1):
                bases[k] = break_base
                n_subs += 1
        i = j
    return "".join(bases), n_subs


# --------------------------------------------------------------------------
# Distance validation
# --------------------------------------------------------------------------

def dna_hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def compute_spatial_distance_stats(
    barcodes: List[str],
    block_rows: int,
    block_cols: int,
) -> Dict[str, Any]:
    """
    Compute Hamming distance statistics between spatially adjacent and
    non-adjacent block pairs.
    """
    adjacent_distances = []
    far_distances = []

    for br in range(block_rows):
        for bc in range(block_cols):
            idx = br * block_cols + bc
            seq = barcodes[idx]

            # Check right neighbour
            if bc + 1 < block_cols:
                n_idx = br * block_cols + (bc + 1)
                d = dna_hamming(seq, barcodes[n_idx])
                adjacent_distances.append(d)

            # Check bottom neighbour
            if br + 1 < block_rows:
                n_idx = (br + 1) * block_cols + bc
                d = dna_hamming(seq, barcodes[n_idx])
                adjacent_distances.append(d)

            # Sample far neighbours (block-distance >= 5)
            for bc2 in range(bc + 5, min(bc + 15, block_cols)):
                n_idx = br * block_cols + bc2
                far_distances.append(dna_hamming(seq, barcodes[n_idx]))
            for br2 in range(br + 5, min(br + 15, block_rows)):
                n_idx = br2 * block_cols + bc
                far_distances.append(dna_hamming(seq, barcodes[n_idx]))

    stats: Dict[str, Any] = {}
    if adjacent_distances:
        stats["adjacent_min"] = min(adjacent_distances)
        stats["adjacent_mean"] = round(sum(adjacent_distances) / len(adjacent_distances), 2)
        stats["adjacent_max"] = max(adjacent_distances)
    if far_distances:
        stats["far_min"] = min(far_distances)
        stats["far_mean"] = round(sum(far_distances) / len(far_distances), 2)
        stats["far_max"] = max(far_distances)

    return stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "block_zipcode_config.json"),
        help="Path to Zipcode config JSON (default: block_zipcode_config.json next to script)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR),
        help="Output directory (default: same directory as this script)",
    )
    ap.add_argument(
        "--no-g-filter",
        action="store_true",
        help="Skip G-content post-filter (for comparison)",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load config and derive block-grid dimensions
    # ------------------------------------------------------------------
    print(f"Loading config from {config_path}", flush=True)
    config = load_config(config_path)

    # Block grid dimensions come from the config's chip_geometry
    block_rows: int = config.chip_geometry.rows
    block_cols: int = config.chip_geometry.cols
    n_blocks: int = block_rows * block_cols
    print(f"Block grid: {block_rows}×{block_cols} = {n_blocks:,} blocks")

    print(f"\n[Step 1] Generating X axis codes ({block_cols} blocks)...")
    x_codes, x_verif = generate_axis_codes(
        n_positions=block_cols,
        bit_length=config.x_bit_length,
        target_distances=config.error_correction.distances,
        far_features_min_distance=config.error_correction.far_features_min_distance,
        max_attempts_per_position=config.axis_code.max_attempts_per_position,
        backtrack_depth=config.axis_code.backtrack_depth,
        tolerance=config.axis_code.tolerance,
        random_seed=config.axis_code.random_seed,
    )
    print(f"  Generated {len(x_codes)} X codes ({len(x_codes[0])} bits each)")
    print(f"  Hard constraint violations: {x_verif['hard_constraint_violations']}")

    print(f"\n[Step 2] Generating Y axis codes ({block_rows} blocks)...")
    y_codes, y_verif = generate_axis_codes(
        n_positions=block_rows,
        bit_length=config.y_bit_length,
        target_distances=config.error_correction.distances,
        far_features_min_distance=config.error_correction.far_features_min_distance,
        max_attempts_per_position=config.axis_code.max_attempts_per_position,
        backtrack_depth=config.axis_code.backtrack_depth,
        tolerance=config.axis_code.tolerance,
        random_seed=(config.axis_code.random_seed + 1
                     if config.axis_code.random_seed is not None
                     else None),
    )
    print(f"  Generated {len(y_codes)} Y codes ({len(y_codes[0])} bits each)")
    print(f"  Hard constraint violations: {y_verif['hard_constraint_violations']}")

    # ------------------------------------------------------------------
    # 2. Encode axis codes to DNA
    # ------------------------------------------------------------------
    print("\n[Step 3] Encoding to DNA sequences...")
    encoder = BitToDNAEncoder(
        scheme=config.bit_to_dna.scheme,
        mapping=config.bit_to_dna.mapping,
    )
    x_seqs = [encoder.encode(code) for code in x_codes]
    y_seqs = [encoder.encode(code) for code in y_codes]
    print(f"  X sequences: {len(x_seqs[0])} nt each")
    print(f"  Y sequences: {len(y_seqs[0])} nt each")

    # ------------------------------------------------------------------
    # 3. Combine X + Y into bar_1 barcodes (no spacer)
    # ------------------------------------------------------------------
    print("\n[Step 4] Combining X + Y into 28 nt bar_1 barcodes...")
    barcodes: List[str] = []
    for br in range(block_rows):
        for bc in range(block_cols):
            bar1 = x_seqs[bc] + y_seqs[br]  # 14nt + 14nt = 28nt
            barcodes.append(bar1)

    assert len(barcodes) == n_blocks, f"Expected {n_blocks} barcodes, got {len(barcodes)}"
    assert all(len(b) == BAR1_LEN for b in barcodes), "Not all barcodes are 28 nt"

    # ------------------------------------------------------------------
    # 4. G-content post-filter
    # ------------------------------------------------------------------
    n_modified = 0
    if not args.no_g_filter:
        print(f"\n[Step 5] G-content post-filter (G <= {G_MAX_FRACTION:.0%} per barcode)...")
        for i, bc_seq in enumerate(barcodes):
            new_seq, n_swaps = reduce_g_content(bc_seq, G_MAX_FRACTION)
            if n_swaps:
                barcodes[i] = new_seq
                n_modified += 1
        print(f"  Modified {n_modified}/{n_blocks} barcodes "
              f"({n_modified / n_blocks:.1%} had G > {G_MAX_FRACTION:.0%})")
    else:
        print("\n[Step 5] G-content post-filter skipped (--no-g-filter)")

    # ------------------------------------------------------------------
    # 5b. Homopolymer post-filter (applied to full 28nt bar_1)
    # ------------------------------------------------------------------
    max_homo_cfg = config.sequence_qc.max_homopolymer_length
    n_hp_fixed = 0
    print(f"\n[Step 5b] Homopolymer post-filter (max run <= {max_homo_cfg} nt)...")
    for i, bc_seq in enumerate(barcodes):
        if max_homopolymer(bc_seq) > max_homo_cfg:
            new_seq, n_subs = break_homopolymers(bc_seq, max_run=max_homo_cfg)
            barcodes[i] = new_seq
            n_hp_fixed += 1
    n_still_bad = sum(1 for b in barcodes if max_homopolymer(b) > max_homo_cfg)
    print(f"  Fixed {n_hp_fixed} barcodes with homopolymer > {max_homo_cfg}")
    if n_still_bad:
        print(f"  WARNING: {n_still_bad} barcodes still exceed limit after fix",
              file=sys.stderr)

    # ------------------------------------------------------------------
    # 6. Global uniqueness check
    # ------------------------------------------------------------------
    print("\n[Step 6] Validating global uniqueness...")
    if len(set(barcodes)) != n_blocks:
        duplicates = n_blocks - len(set(barcodes))
        print(f"  WARNING: {duplicates} duplicate barcodes found!", file=sys.stderr)
    else:
        print(f"  OK — all {n_blocks} bar_1 barcodes are globally unique")

    # ------------------------------------------------------------------
    # 6. Distance statistics
    # ------------------------------------------------------------------
    print("\n[Step 7] Computing spatial distance statistics...")
    dist_stats = compute_spatial_distance_stats(barcodes, block_rows, block_cols)
    print(f"  Adjacent block Hamming: "
          f"min={dist_stats.get('adjacent_min', '?')}  "
          f"mean={dist_stats.get('adjacent_mean', '?')}  "
          f"max={dist_stats.get('adjacent_max', '?')}")
    print(f"  Far block Hamming (>=5 steps): "
          f"min={dist_stats.get('far_min', '?')}  "
          f"mean={dist_stats.get('far_mean', '?')}  "
          f"max={dist_stats.get('far_max', '?')}")

    # ------------------------------------------------------------------
    # 7. Per-barcode QC summary
    # ------------------------------------------------------------------
    g_values = [g_fraction(b) for b in barcodes]
    gc_values = [gc_fraction(b) for b in barcodes]
    hp_values = [max_homopolymer(b) for b in barcodes]
    n_g_fail = sum(1 for g in g_values if g > G_MAX_FRACTION)
    print(f"\n  G content:  mean={sum(g_values)/len(g_values):.3f}  "
          f"max={max(g_values):.3f}  >25%: {n_g_fail}")
    print(f"  GC content: mean={sum(gc_values)/len(gc_values):.3f}  "
          f"range [{min(gc_values):.3f}, {max(gc_values):.3f}]")
    print(f"  Max homopolymer: {max(hp_values)}")

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    # Plain text: one barcode per line, row-major (block_row 0 first)
    txt_path = out_dir / "bar1_barcodes_28nt.txt"
    with open(txt_path, "w") as fh:
        for seq in barcodes:
            fh.write(seq + "\n")
    print(f"\nWrote barcodes       → {txt_path}")

    # CSV with block coordinates
    csv_path = out_dir / "bar1_codebook.csv"
    with open(csv_path, "w") as fh:
        fh.write("block_row,block_col,bar_1\n")
        for br in range(block_rows):
            for bc in range(block_cols):
                idx = br * block_cols + bc
                fh.write(f"{br},{bc},{barcodes[idx]}\n")
    print(f"Wrote codebook CSV   → {csv_path}")

    # QC report
    qc_path = out_dir / "bar1_qc_report.json"
    qc_report = {
        "n_blocks": n_blocks,
        "block_grid": f"{block_rows}x{block_cols}",
        "bar1_length_nt": BAR1_LEN,
        "n_globally_unique": len(set(barcodes)),
        "g_filter_applied": not args.no_g_filter,
        "n_modified_by_g_filter": n_modified,
        "g_content": {
            "mean": round(sum(g_values) / len(g_values), 4),
            "max": round(max(g_values), 4),
            "n_above_25pct": n_g_fail,
        },
        "gc_content": {
            "mean": round(sum(gc_values) / len(gc_values), 4),
            "min": round(min(gc_values), 4),
            "max": round(max(gc_values), 4),
        },
        "max_homopolymer": int(max(hp_values)),
        "spatial_hamming_distance": dist_stats,
        "axis_code_generation": {
            "x_hard_violations": x_verif["hard_constraint_violations"],
            "y_hard_violations": y_verif["hard_constraint_violations"],
        },
    }
    with open(qc_path, "w") as fh:
        json.dump(qc_report, fh, indent=2)
    print(f"Wrote QC report      → {qc_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
