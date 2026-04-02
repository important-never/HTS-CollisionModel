#!/usr/bin/env python3
"""
HPC-7: S-layer Shuffle Causality Test
======================================
ONE task = ONE (species, anchor, shuffle_seed).
Shuffles S values within species to break enzyme-substrate specificity,
then recomputes RRI. If shuffle destroys RRI → proves S-layer causality.

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
    score_all_pairs, compute_delta_rank, compute_rri,
    setup_task_logger, to_csv_safe, S_NEUTRAL,
)


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_{args.anchor}_seed{args.seed}"
    log_dir = out_dir.parent / "logs" / "hpc7"
    log = setup_task_logger("hpc7", log_dir, tag)

    log.info(f"S-shuffle causality: {args.species} / {args.anchor} / seed={args.seed}")

    master = load_master(args.data_dir)
    s_dict = load_s_values(args.data_dir)
    known = load_known_pairs(args.data_dir)

    sp = args.species
    anchor = args.anchor
    known_set = known.get(sp, set())

    sp_df = master[master["sp"] == sp]
    mets = sp_df[sp_df["Category"] == "Metabolite"]
    prots = sp_df[sp_df["Category"] == "Protein"]

    anchor_rows = mets[mets["Node_Name"] == anchor]
    if len(anchor_rows) == 0:
        log.warning(f"Anchor {anchor} not found in {sp}")
        return
    anchor_row = anchor_rows.iloc[0]

    # ── Baseline (real S) ──
    j_base, tl_base, names = score_all_pairs(
        anchor_row, prots, s_dict, sp,
    )
    dr_base = compute_delta_rank(j_base, tl_base)
    known_mask = np.array([(anchor, n) in known_set or (n, anchor) in known_set for n in names])
    rri_base = compute_rri(dr_base, known_mask)

    # ── Shuffle S values within species ──
    rng = np.random.RandomState(args.seed)

    # Collect all S keys for this species
    sp_keys = [k for k in s_dict if len(k) == 3 and k[2] == sp]
    sp_vals = [s_dict[k] for k in sp_keys]

    # Shuffle values (break enzyme-substrate specificity)
    shuffled_vals = sp_vals.copy()
    rng.shuffle(shuffled_vals)

    # Build shuffled dict
    s_shuffled = dict(s_dict)  # copy
    for k, v in zip(sp_keys, shuffled_vals):
        s_shuffled[k] = v

    # Score with shuffled S
    j_shuf, tl_shuf, names_shuf = score_all_pairs(
        anchor_row, prots, s_shuffled, sp,
    )
    dr_shuf = compute_delta_rank(j_shuf, tl_shuf)
    rri_shuf = compute_rri(dr_shuf, known_mask)

    n_known = int(known_mask.sum())
    pos_ratio_base = float((dr_base[known_mask] > 0).mean()) if n_known > 0 else np.nan
    pos_ratio_shuf = float((dr_shuf[known_mask] > 0).mean()) if n_known > 0 else np.nan

    row = {
        "species": sp,
        "anchor": anchor,
        "seed": args.seed,
        "n_known": n_known,
        "rri_real": rri_base,
        "rri_shuffled": rri_shuf,
        "rri_delta": rri_base - rri_shuf,
        "pos_ratio_real": pos_ratio_base,
        "pos_ratio_shuffled": pos_ratio_shuf,
    }

    df = pd.DataFrame([row])
    csv_path = out_dir / f"hpc7_{tag}.csv"
    to_csv_safe(df, csv_path)
    log.info(f"Done: RRI_real={rri_base:.2f}, RRI_shuffled={rri_shuf:.2f}, delta={rri_base - rri_shuf:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--species", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_dir", default="results/hpc7")
    args = ap.parse_args()

    tag = f"{args.species}_{args.anchor}_seed{args.seed}"
    try:
        run(args)
    except Exception:
        err_dir = Path(args.output_dir).parent / "logs" / "hpc7"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
