#!/usr/bin/env python3
"""
Phase 1 — Generate 16 spot barcodes (bar_2) for the HSDB design.

Selects 16 barcodes of 8 nt with:
  - Minimum pairwise Hamming distance >= MIN_HAMMING (target >= 5, fallback to 4)
  - GC content 40–60%
  - G content <= 20%  (max 2 G's in 8 nt, per Idea 3 constraint)
  - No homopolymer run > 3 nt

These 16 barcodes are the fixed bar_2 set, reused in every block.
Global uniqueness of each spot is guaranteed by the unique bar_1 of its block.

Outputs: spot_barcodes_8nt.txt (one sequence per line)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import List, Tuple

BARCODE_LEN = 8
N_BARCODES = 16
BASES = "ACGT"


# --------------------------------------------------------------------------
# Sequence QC helpers
# --------------------------------------------------------------------------

def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def g_fraction(seq: str) -> float:
    return seq.count("G") / len(seq)


def max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    max_run = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run


def passes_qc(seq: str,
              gc_min: float = 0.40,
              gc_max: float = 0.60,
              g_max: float = 0.20,
              max_homo: int = 3) -> bool:
    return (gc_min <= gc_content(seq) <= gc_max
            and g_fraction(seq) <= g_max
            and max_homopolymer(seq) <= max_homo)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def min_pairwise_hamming(seqs: List[str]) -> int:
    if len(seqs) < 2:
        return 0
    return min(hamming(a, b)
               for i, a in enumerate(seqs)
               for b in seqs[i + 1:])


# --------------------------------------------------------------------------
# Greedy selection
# --------------------------------------------------------------------------

def enumerate_candidates(length: int = BARCODE_LEN) -> List[str]:
    """Return all k-mers that pass QC filters."""
    return [
        "".join(b)
        for b in itertools.product(BASES, repeat=length)
        if passes_qc("".join(b))
    ]


def greedy_select(candidates: List[str],
                  n: int,
                  min_dist: int) -> List[str]:
    """
    Greedy set packing: iteratively add a candidate sequence that is at
    least min_dist (Hamming) from every already-selected sequence.
    """
    selected: List[str] = []
    for cand in candidates:
        if all(hamming(cand, s) >= min_dist for s in selected):
            selected.append(cand)
            if len(selected) == n:
                break
    return selected


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=str(Path(__file__).parent / "spot_barcodes_8nt.txt"),
                    help="Output path for the selected barcodes (default: same dir as script)")
    ap.add_argument("--min-hamming", type=int, default=5,
                    help="Target minimum pairwise Hamming distance (default 5; "
                         "falls back to 4 if not achievable)")
    ap.add_argument("--n", type=int, default=N_BARCODES,
                    help=f"Number of spot barcodes to select (default {N_BARCODES})")
    ap.add_argument("--length", type=int, default=BARCODE_LEN,
                    help=f"Barcode length in nt (default {BARCODE_LEN})")
    ap.add_argument("--stats-out", default=None,
                    help="Optional JSON file to write per-barcode statistics")
    args = ap.parse_args()

    print(f"Enumerating all {args.length}-mers passing QC...", flush=True)
    candidates = enumerate_candidates(args.length)
    print(f"  {len(candidates):,} candidates pass QC "
          f"(GC 40-60%, G<=25%, homopolymer<=3)")

    # Try from requested min_hamming down to 4
    selected: List[str] = []
    achieved_dist = 0
    for min_hd in range(args.min_hamming, 3, -1):
        selected = greedy_select(candidates, args.n, min_hd)
        if len(selected) == args.n:
            achieved_dist = min_hd
            break
        print(f"  Only {len(selected)}/{args.n} found at min_dist={min_hd}, "
              f"retrying with {min_hd - 1}...")

    if len(selected) < args.n:
        print(f"ERROR: could not select {args.n} barcodes with min Hamming >= 4",
              file=sys.stderr)
        sys.exit(1)

    actual_min = min_pairwise_hamming(selected)

    # ---------- Report ---------
    print(f"\n  Selected {len(selected)} barcodes with min Hamming distance = {actual_min}")
    print("\n  #   Sequence    GC    G     MaxHP")
    print("  " + "-" * 37)
    stats_rows = []
    for i, seq in enumerate(selected):
        gc = gc_content(seq)
        gf = g_fraction(seq)
        hp = max_homopolymer(seq)
        print(f"  {i + 1:2d}  {seq}   {gc:.2f}  {gf:.2f}  {hp}")
        stats_rows.append({
            "idx": i,
            "seq": seq,
            "gc_content": round(gc, 4),
            "g_fraction": round(gf, 4),
            "max_homopolymer": hp,
        })

    print(f"\n  Min pairwise Hamming distance: {actual_min}")
    print(f"  GC content range: "
          f"{min(r['gc_content'] for r in stats_rows):.2f} – "
          f"{max(r['gc_content'] for r in stats_rows):.2f}")
    print(f"  Max homopolymer : "
          f"{max(r['max_homopolymer'] for r in stats_rows)}")

    # ---------- Write barcodes ---------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for seq in selected:
            fh.write(seq + "\n")
    print(f"\nWrote {len(selected)} barcodes → {out_path}")

    # ---------- Optionally write stats JSON ---------
    if args.stats_out:
        stats_path = Path(args.stats_out)
        with open(stats_path, "w") as fh:
            json.dump({
                "n_barcodes": len(selected),
                "barcode_length": args.length,
                "min_hamming_distance": actual_min,
                "barcodes": stats_rows,
            }, fh, indent=2)
        print(f"Wrote stats        → {stats_path}")


if __name__ == "__main__":
    main()
