#!/usr/bin/env python3
"""
Phase 4 — Benchmark HSDB vs current Dual Barcodes (18+18).

Simulates sequencing reads with the Press-style error model
(sub=0.03, ins=0.04, del=0.14) and evaluates two layouts:

  A) Current Dual Barcodes  — 18 nt bar_1 + 18 nt bar_2 = 36 nt
  B) HSDB flat              — 28 nt bar_1 + 8 nt bar_2 = 36 nt (random bar_1)
  C) HSDB spatial           — same as B but bar_1 from Zipcode block grid

Two decoders are compared:
  1. Flat decoder       — nearest-neighbour over all 21,476 combined 36 nt barcodes
  2. Hierarchical decoder — decode bar_1 (1-of-1,380), then bar_2 (1-of-16)

Metrics reported:
  Precision = n_correct_decoded / n_decoded
  Recall    = n_decoded / n_total_reads
  Erasure rate = n_erased / n_total_reads

Usage:
  python benchmark.py --hsdb-barcodes outputs/barcodes_hsdb_36nt.txt \\
                      --dual-barcodes ../Dual_barcodes/barcodes_118x182_dual_set1.txt \\
                      --reads 250000 --seed 1

  Or run all available layouts in one call (default, uses built-in paths).
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Default paths
DEFAULT_HSDB_BARCODES = REPO_ROOT / "HSDB" / "outputs" / "barcodes_hsdb_36nt.txt"
DEFAULT_DUAL_BARCODES = (REPO_ROOT / "Dual_barcodes" /
                         "barcodes_118x182_dual_set1.txt")
DEFAULT_HSDB_BAR1 = REPO_ROOT / "HSDB" / "bar1_zipcode" / "bar1_barcodes_28nt.txt"
DEFAULT_HSDB_BAR2 = REPO_ROOT / "HSDB" / "bar2_library" / "spot_barcodes_8nt.txt"

# Error model parameters (matching the current benchmark)
DEFAULT_SRATE = 0.03
DEFAULT_IRATE = 0.04
DEFAULT_DRATE = 0.14
DEFAULT_N_READS = 250_000
DEFAULT_MAX_DIST_FLAT = 5       # Levenshtein threshold for flat decoder
DEFAULT_MAX_DIST_BAR1 = 4       # Levenshtein threshold for hierarchical bar_1
DEFAULT_MAX_DIST_BAR2 = 2       # Levenshtein threshold for hierarchical bar_2

BASES = ["A", "C", "G", "T"]
BARCODE_LEN = 36
BAR1_LEN = 28
BAR2_LEN = 8


# ==========================================================================
# Error model (Press-style, no separator)
# ==========================================================================

def binomial_py(rng: random.Random, n: int, p: float) -> int:
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)


def apply_errors(seq: str,
                 srate: float,
                 irate: float,
                 drate: float,
                 rng: random.Random) -> str:
    """
    Apply substitutions, deletions, insertions uniformly across all positions.
    Then pad / truncate to restore original length.

    This is the simple form without separator protection — correct for
    the HSDB layout which has no embedded separator.
    """
    chars = list(seq.upper())
    n = len(chars)

    # Substitutions: Binomial(n, (4/3)*srate) positions, assign random base
    p_sub = min(1.0, (4.0 / 3.0) * srate)
    n_sub = binomial_py(rng, n, p_sub)
    if n_sub > 0:
        positions = [rng.randrange(n) for _ in range(n_sub)]
        for pos in positions:
            chars[pos] = rng.choice(BASES)

    # Deletions: Binomial(n, drate) positions removed
    n_del = binomial_py(rng, len(chars), drate)
    if n_del > 0:
        chosen = sorted(set(rng.randrange(len(chars)) for _ in range(n_del)),
                        reverse=True)
        for pos in chosen:
            del chars[pos]

    # Insertions: Binomial(len, irate) random bases inserted
    n_ins = binomial_py(rng, len(chars), irate)
    if n_ins > 0:
        for _ in range(n_ins):
            pos = rng.randrange(len(chars) + 1)
            chars.insert(pos, rng.choice(BASES))

    # Pad or truncate to original length
    while len(chars) < n:
        chars.append(rng.choice(BASES))
    chars = chars[:n]

    return "".join(chars)


# ==========================================================================
# Levenshtein distance (dynamic programming, capped at max_dist for speed)
# ==========================================================================

def levenshtein(a: str, b: str, cap: int = 999) -> int:
    """Levenshtein distance, capped at `cap` for early exit."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    # Use a single-row DP
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,          # deletion
                curr[j] + 1,              # insertion
                prev[j] + (ca != cb),     # substitution
            )
            if curr[j + 1] > cap:
                curr[j + 1] = cap + 1
        prev = curr
    return prev[lb]


# ==========================================================================
# Decoders
# ==========================================================================

def flat_decode(read: str,
                barcodes: List[str],
                max_dist: int) -> Tuple[Optional[int], int]:
    """
    Nearest-neighbour Levenshtein decoder over the full barcode library.
    Returns (best_index, best_distance) or (None, best_distance) if erased.
    """
    best_dist = max_dist + 1
    best_idx = None
    n_best = 0

    for i, ref in enumerate(barcodes):
        d = levenshtein(read, ref, cap=best_dist)
        if d < best_dist:
            best_dist = d
            best_idx = i
            n_best = 1
        elif d == best_dist:
            n_best += 1

    # Erasure if no match or tie (ambiguous)
    if best_idx is None or n_best > 1:
        return None, best_dist
    return best_idx, best_dist


def hierarchical_decode(read: str,
                        bar1_lib: List[str],
                        bar2_lib: List[str],
                        max_dist_bar1: int,
                        max_dist_bar2: int) -> Tuple[Optional[int], Optional[int], int, int]:
    """
    Two-stage decoder:
      Stage 1: decode first BAR1_LEN bases against bar_1 library (1-of-1,380)
      Stage 2: decode last BAR2_LEN bases against bar_2 library (1-of-16)

    Returns (bar1_idx, bar2_idx, bar1_dist, bar2_dist).
    bar1_idx or bar2_idx is None on erasure / tie.
    """
    obs_bar1 = read[:BAR1_LEN]
    obs_bar2 = read[BAR1_LEN:]

    # Stage 1
    best1, best1_idx, n_best1 = max_dist_bar1 + 1, None, 0
    for i, ref in enumerate(bar1_lib):
        d = levenshtein(obs_bar1, ref, cap=best1)
        if d < best1:
            best1 = d
            best1_idx = i
            n_best1 = 1
        elif d == best1:
            n_best1 += 1
    if best1_idx is None or n_best1 > 1:
        best1_idx = None

    # Stage 2
    best2, best2_idx, n_best2 = max_dist_bar2 + 1, None, 0
    for i, ref in enumerate(bar2_lib):
        d = levenshtein(obs_bar2, ref, cap=best2)
        if d < best2:
            best2 = d
            best2_idx = i
            n_best2 = 1
        elif d == best2:
            n_best2 += 1
    if best2_idx is None or n_best2 > 1:
        best2_idx = None

    return best1_idx, best2_idx, best1, best2


# ==========================================================================
# Simulation runner
# ==========================================================================

def simulate(
    barcodes: List[str],
    n_reads: int,
    srate: float,
    irate: float,
    drate: float,
    seed: int,
    decoder: str,                   # "flat" | "hierarchical"
    max_dist_flat: int,
    max_dist_bar1: int,
    max_dist_bar2: int,
    bar1_lib: Optional[List[str]] = None,
    bar2_lib: Optional[List[str]] = None,
    label: str = "",
) -> Dict:
    """
    Run a full simulation and return a results dict.

    For 'flat' decoder, barcodes must be the full 36 nt combined library.
    For 'hierarchical', bar1_lib and bar2_lib must be provided.
    The truth mapping: barcode index → (bar1_idx, bar2_idx) is derived from
    the order of barcodes (row-major: block 0 bar2_0, block 0 bar2_1, ...).
    """
    rng = random.Random(seed)
    n_barcodes = len(barcodes)

    n_correct = 0
    n_decoded = 0
    n_erased = 0
    n_wrong = 0

    progress_step = max(1, n_reads // 20)

    print(f"  [{label}] Simulating {n_reads:,} reads with "
          f"decoder='{decoder}' ...", flush=True)

    for read_i in range(n_reads):
        if (read_i + 1) % progress_step == 0:
            pct = 100 * (read_i + 1) / n_reads
            print(f"    {pct:.0f}%", end="\r", flush=True)

        # Sample a random true barcode
        true_idx = rng.randrange(n_barcodes)
        true_barcode = barcodes[true_idx]

        # Apply errors
        read = apply_errors(true_barcode, srate, irate, drate, rng)

        # Decode
        if decoder == "flat":
            pred_idx, _ = flat_decode(read, barcodes, max_dist_flat)
            erased = pred_idx is None
            correct = (not erased) and (pred_idx == true_idx)

        elif decoder == "hierarchical":
            assert bar1_lib is not None and bar2_lib is not None
            # True bar_1 and bar_2 indices: derived from barcode order
            # In HSDB layout: barcodes are in row-major (spot) order;
            # the bar_1 of barcode at index i is barcodes[i][:BAR1_LEN].
            # The hierarchical decoder returns block and spot indices.
            true_bar1 = true_barcode[:BAR1_LEN]
            true_bar2 = true_barcode[BAR1_LEN:]

            # Map true sequences to library indices
            true_bar1_idx = bar1_lib.index(true_bar1) if true_bar1 in bar1_lib else -1
            true_bar2_idx = bar2_lib.index(true_bar2) if true_bar2 in bar2_lib else -1

            pred1, pred2, _, _ = hierarchical_decode(
                read, bar1_lib, bar2_lib, max_dist_bar1, max_dist_bar2
            )
            erased = (pred1 is None) or (pred2 is None)
            correct = (not erased) and (pred1 == true_bar1_idx) and (pred2 == true_bar2_idx)
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        if erased:
            n_erased += 1
        elif correct:
            n_correct += 1
            n_decoded += 1
        else:
            n_wrong += 1
            n_decoded += 1

    print(f"    100%", flush=True)

    recall = n_decoded / n_reads if n_reads > 0 else 0.0
    precision = n_correct / n_decoded if n_decoded > 0 else 0.0
    erasure_rate = n_erased / n_reads if n_reads > 0 else 0.0

    return {
        "label": label,
        "decoder": decoder,
        "n_reads": n_reads,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_erased": n_erased,
        "n_decoded": n_decoded,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "erasure_rate": round(erasure_rate, 6),
        "error_rates": {"sub": srate, "ins": irate, "del": drate},
    }


# ==========================================================================
# I/O
# ==========================================================================

def load_barcodes(path: Path, expected_len: Optional[int] = None) -> List[str]:
    seqs = [line.strip().upper() for line in open(path) if line.strip()]
    if not seqs:
        raise FileNotFoundError(f"No sequences in {path}")
    if expected_len:
        wrong = [s for s in seqs if len(s) != expected_len]
        if wrong:
            print(f"  WARNING: {len(wrong)} sequences in {path} have "
                  f"unexpected length (expected {expected_len})")
    return seqs


def print_results_table(results: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Layout / Decoder':<38} {'Precision':>10} {'Recall':>9} {'Erasure%':>9}")
    print("-" * 72)
    for r in results:
        lbl = f"{r['label']} [{r['decoder']}]"
        print(f"  {lbl:<36} {r['precision']:>10.4f} {r['recall']:>9.4f} "
              f"{r['erasure_rate'] * 100:>8.2f}%")
    print("=" * 72)


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hsdb-barcodes", default=str(DEFAULT_HSDB_BARCODES),
                    help="HSDB combined 36 nt barcodes file")
    ap.add_argument("--hsdb-bar1", default=str(DEFAULT_HSDB_BAR1),
                    help="HSDB bar_1 library (28 nt)")
    ap.add_argument("--hsdb-bar2", default=str(DEFAULT_HSDB_BAR2),
                    help="HSDB bar_2 library (8 nt)")
    ap.add_argument("--dual-barcodes", default=str(DEFAULT_DUAL_BARCODES),
                    help="Current Dual Barcodes 36 nt file (for comparison)")
    ap.add_argument("--reads", type=int, default=DEFAULT_N_READS,
                    help=f"Total reads to simulate (default {DEFAULT_N_READS:,})")
    ap.add_argument("--srate", type=float, default=DEFAULT_SRATE)
    ap.add_argument("--irate", type=float, default=DEFAULT_IRATE)
    ap.add_argument("--drate", type=float, default=DEFAULT_DRATE)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-dist-flat", type=int, default=DEFAULT_MAX_DIST_FLAT,
                    help="Max Levenshtein distance for flat decoder")
    ap.add_argument("--max-dist-bar1", type=int, default=DEFAULT_MAX_DIST_BAR1,
                    help="Max Levenshtein distance for hierarchical bar_1 stage")
    ap.add_argument("--max-dist-bar2", type=int, default=DEFAULT_MAX_DIST_BAR2,
                    help="Max Levenshtein distance for hierarchical bar_2 stage")
    ap.add_argument("--skip-flat", action="store_true",
                    help="Skip flat decoder (much faster, only run hierarchical)")
    ap.add_argument("--skip-dual", action="store_true",
                    help="Skip current Dual Barcodes baseline")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "benchmark_results.json"),
                    help="Output JSON file for results")
    args = ap.parse_args()

    all_results = []
    error_kw = dict(srate=args.srate, irate=args.irate, drate=args.drate)

    # ------------------------------------------------------------------
    # 1. Current Dual Barcodes — flat decoder (baseline)
    # ------------------------------------------------------------------
    dual_path = Path(args.dual_barcodes)
    if not args.skip_dual and dual_path.exists():
        print(f"\nLoading Dual Barcodes baseline from {dual_path}...")
        dual_barcodes = load_barcodes(dual_path, expected_len=BARCODE_LEN)
        print(f"  {len(dual_barcodes):,} barcodes loaded")

        if not args.skip_flat:
            r = simulate(
                barcodes=dual_barcodes,
                n_reads=args.reads,
                seed=args.seed,
                decoder="flat",
                max_dist_flat=args.max_dist_flat,
                max_dist_bar1=args.max_dist_bar1,
                max_dist_bar2=args.max_dist_bar2,
                label="Dual-18+18 baseline",
                **error_kw,
            )
            all_results.append(r)
    elif not dual_path.exists():
        print(f"\nSkipping Dual Barcodes baseline: file not found ({dual_path})")

    # ------------------------------------------------------------------
    # 2. HSDB — flat decoder
    # ------------------------------------------------------------------
    hsdb_path = Path(args.hsdb_barcodes)
    hsdb_bar1_path = Path(args.hsdb_bar1)
    hsdb_bar2_path = Path(args.hsdb_bar2)

    hsdb_barcodes: Optional[List[str]] = None
    bar1_lib: Optional[List[str]] = None
    bar2_lib: Optional[List[str]] = None

    if hsdb_path.exists():
        print(f"\nLoading HSDB combined barcodes from {hsdb_path}...")
        hsdb_barcodes = load_barcodes(hsdb_path, expected_len=BARCODE_LEN)
        print(f"  {len(hsdb_barcodes):,} barcodes loaded")

        if not args.skip_flat:
            r = simulate(
                barcodes=hsdb_barcodes,
                n_reads=args.reads,
                seed=args.seed,
                decoder="flat",
                max_dist_flat=args.max_dist_flat,
                max_dist_bar1=args.max_dist_bar1,
                max_dist_bar2=args.max_dist_bar2,
                label="HSDB-28+8 spatial",
                **error_kw,
            )
            all_results.append(r)
    else:
        print(f"\nSkipping HSDB flat benchmark: file not found ({hsdb_path})")
        print("  Run Phase 3 (layout_generator/dual_barcode_hierarchical.py) first.")

    # ------------------------------------------------------------------
    # 3. HSDB — hierarchical decoder
    # ------------------------------------------------------------------
    if hsdb_barcodes is not None and hsdb_bar1_path.exists() and hsdb_bar2_path.exists():
        print(f"\nLoading bar_1 library from {hsdb_bar1_path}...")
        bar1_lib = load_barcodes(hsdb_bar1_path, expected_len=BAR1_LEN)
        print(f"  {len(bar1_lib)} bar_1 sequences loaded")

        print(f"Loading bar_2 library from {hsdb_bar2_path}...")
        bar2_lib = load_barcodes(hsdb_bar2_path, expected_len=BAR2_LEN)
        print(f"  {len(bar2_lib)} bar_2 sequences loaded")

        r = simulate(
            barcodes=hsdb_barcodes,
            n_reads=args.reads,
            seed=args.seed,
            decoder="hierarchical",
            max_dist_flat=args.max_dist_flat,
            max_dist_bar1=args.max_dist_bar1,
            max_dist_bar2=args.max_dist_bar2,
            bar1_lib=bar1_lib,
            bar2_lib=bar2_lib,
            label="HSDB-28+8 spatial",
            **error_kw,
        )
        all_results.append(r)
    else:
        missing = [p for p in [hsdb_bar1_path, hsdb_bar2_path] if not p.exists()]
        if missing:
            print(f"\nSkipping hierarchical decoder: missing files: {missing}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if not all_results:
        print("\nNo results to report. Run Phase 1–3 first to generate barcodes.",
              file=sys.stderr)
        sys.exit(1)

    print_results_table(all_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
