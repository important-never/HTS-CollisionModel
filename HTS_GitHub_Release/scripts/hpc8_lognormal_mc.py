#!/usr/bin/env python3
"""
HPC-8: Log-Normal Noise Monte Carlo (Breaking Point Finder)
============================================================
ONE task = ONE (species, anchor, sigma, seed).
Applies log-normal noise to S values (not linear noise like HPC-2),
capable of perturbing across orders of magnitude to find the breaking point.

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
    setup_task_logger, to_csv_safe,
)

N_ITER = 200  # iterations per task


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_{args.anchor}_sigma{args.sigma:.1f}_s{args.seed}"
    log_dir = out_dir.parent / "logs" / "hpc8"
    log = setup_task_logger("hpc8", log_dir, tag)

    log.info(f"Log-normal MC: {args.species}/{args.anchor} sigma={args.sigma} seed={args.seed}")

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

    # ── Baseline ──
    j_base, tl_base, names = score_all_pairs(anchor_row, prots, s_dict, sp)
    dr_base = compute_delta_rank(j_base, tl_base)
    known_mask = np.array([(anchor, n) in known_set or (n, anchor) in known_set for n in names])
    rri_base = compute_rri(dr_base, known_mask)
    n_known = int(known_mask.sum())
    n_partners = len(names)

    if n_known == 0:
        row = {
            "species": sp, "anchor": anchor, "sigma": args.sigma,
            "seed": args.seed, "n_known": 0, "n_partners": n_partners,
            "baseline_rri": rri_base, "mean_rri": np.nan, "std_rri": np.nan,
            "flip_probability": np.nan, "mean_pos_ratio": np.nan,
        }
        to_csv_safe(pd.DataFrame([row]), out_dir / f"hpc8_{tag}.csv")
        return

    # ── Monte Carlo with log-normal noise on S ──
    rng = np.random.RandomState(args.seed)
    rri_list = []
    flip_count = 0
    pos_ratios = []

    for _ in range(N_ITER):
        # Log-normal noise: multiply S by 10^(N(0, sigma))
        # This perturbs S across orders of magnitude
        noise_s = np.power(10.0, rng.normal(0, args.sigma, size=n_partners))

        _, tl_noisy, _ = score_all_pairs(
            anchor_row, prots, s_dict, sp, noise_s=noise_s,
        )
        dr_noisy = compute_delta_rank(j_base, tl_noisy)
        rri_noisy = compute_rri(dr_noisy, known_mask)
        rri_list.append(rri_noisy)

        if rri_noisy < 0:
            flip_count += 1
        pr = float((dr_noisy[known_mask] > 0).mean())
        pos_ratios.append(pr)

    rri_arr = np.array(rri_list)
    row = {
        "species": sp,
        "anchor": anchor,
        "sigma": args.sigma,
        "seed": args.seed,
        "n_known": n_known,
        "n_partners": n_partners,
        "baseline_rri": rri_base,
        "mean_rri": float(rri_arr.mean()),
        "std_rri": float(rri_arr.std()),
        "flip_probability": flip_count / N_ITER,
        "mean_pos_ratio": float(np.mean(pos_ratios)),
    }

    df = pd.DataFrame([row])
    to_csv_safe(df, out_dir / f"hpc8_{tag}.csv")
    log.info(f"Done: baseline={rri_base:.2f}, mean={rri_arr.mean():.2f}, "
             f"flip={flip_count}/{N_ITER}, sigma={args.sigma}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--species", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_dir", default="results/hpc8")
    args = ap.parse_args()

    tag = f"{args.species}_{args.anchor}_sigma{args.sigma:.1f}_s{args.seed}"
    try:
        run(args)
    except Exception:
        err_dir = Path(args.output_dir).parent / "logs" / "hpc8"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
