#!/usr/bin/env python3
"""
HPC-1: Genome-wide Rank Rescue Atlas
=====================================
ONE task = ONE (species, anchor_metabolite) pair.
Computes delta_rank for all protein partners, permutation test, outputs CSV.

Error-safe: exceptions logged to independent file; parallel jobs never crash.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_common import (
    load_master, load_s_values, load_known_pairs,
    score_all_pairs, compute_delta_rank, compute_rri, permutation_test_rri,
    setup_task_logger, to_csv_safe,
)


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir.parent / "logs" / "hpc1"
    tag = f"{args.species}_{args.anchor.replace(' ', '_')}"
    log = setup_task_logger("hpc1", log_dir, tag)

    log.info(f"START  species={args.species}  anchor={args.anchor}")

    master = load_master(args.data_dir)
    s_dict = load_s_values(args.data_dir)
    known = load_known_pairs(args.data_dir)

    sp_df = master[master["sp"] == args.species]
    prots = sp_df[sp_df["Category"] == "Protein"]
    mets = sp_df[sp_df["Category"] == "Metabolite"]

    anchor_rows = mets[mets["Node_Name"] == args.anchor]
    if anchor_rows.empty:
        log.warning(f"SKIP — anchor '{args.anchor}' not found in {args.species}")
        return

    anchor_row = anchor_rows.iloc[0]
    rng = np.random.RandomState(args.seed)

    j_scores, tl_scores, partner_names = score_all_pairs(
        anchor_row, prots, s_dict, args.species,
    )
    delta_ranks = compute_delta_rank(j_scores, tl_scores)

    known_set = known.get(args.species, set())
    known_mask = np.array([
        (args.anchor, pn) in known_set or (pn, args.anchor) in known_set
        for pn in partner_names
    ])

    rri = compute_rri(delta_ranks, known_mask)
    p_val = permutation_test_rri(delta_ranks, known_mask, args.n_perm, rng)
    pos_ratio = (
        float(np.mean(delta_ranks[known_mask] > 0))
        if known_mask.sum() > 0 else np.nan
    )

    top_idx = np.argsort(-delta_ranks)[:3]
    top3 = ";".join(
        f"{partner_names[i]}({delta_ranks[i]:.0f})" for i in top_idx
    )

    result = {
        "species": args.species,
        "anchor": args.anchor,
        "n_proteins": len(prots),
        "n_known": int(known_mask.sum()),
        "mean_delta_rank": rri,
        "median_delta_rank": (
            float(np.median(delta_ranks[known_mask]))
            if known_mask.sum() > 0 else np.nan
        ),
        "positive_ratio": pos_ratio,
        "permutation_p": p_val,
        "n_perm": args.n_perm,
        "top3_rescued": top3,
    }

    csv_path = out_dir / f"hpc1_{tag}.csv"
    to_csv_safe(pd.DataFrame([result]), csv_path)
    log.info(
        f"DONE   RRI={rri:.2f}  p={p_val:.4f}  n_known={int(known_mask.sum())}  "
        f"→ {csv_path.name}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    try:
        run(args)
    except Exception:
        tag = f"{args.species}_{args.anchor.replace(' ', '_')}"
        err_dir = Path(args.output_dir).parent / "logs" / "hpc1"
        err_dir.mkdir(parents=True, exist_ok=True)
        err_file = err_dir / f"{tag}.error"
        err_file.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag} — see {err_file}", file=sys.stderr)
        sys.exit(0)  # exit 0 so SLURM array continues


if __name__ == "__main__":
    main()
