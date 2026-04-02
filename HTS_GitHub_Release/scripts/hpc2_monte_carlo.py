#!/usr/bin/env python3
"""
HPC-2: Large-scale Monte Carlo Robustness Scan
================================================
ONE task = ONE (scenario, epsilon, species, anchor) combination.
Error-safe with independent logging.
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


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_{args.anchor}_{args.scenario}_eps{args.epsilon:.2f}_s{args.seed}"
    log_dir = out_dir.parent / "logs" / "hpc2"
    log = setup_task_logger("hpc2", log_dir, tag)

    log.info(f"START  {tag}  n_iter={args.n_iter}")

    master = load_master(args.data_dir)
    s_dict = load_s_values(args.data_dir)
    known = load_known_pairs(args.data_dir)

    sp_df = master[master["sp"] == args.species]
    prots = sp_df[sp_df["Category"] == "Protein"]
    mets = sp_df[sp_df["Category"] == "Metabolite"]

    anchor_rows = mets[mets["Node_Name"] == args.anchor]
    if anchor_rows.empty:
        log.warning(f"SKIP — anchor not found")
        return
    anchor_row = anchor_rows.iloc[0]

    rng = np.random.RandomState(args.seed)
    known_set = known.get(args.species, set())

    j_base, tl_base, partner_names = score_all_pairs(
        anchor_row, prots, s_dict, args.species,
    )
    known_mask = np.array([
        (args.anchor, pn) in known_set or (pn, args.anchor) in known_set
        for pn in partner_names
    ])
    baseline_rri = compute_rri(
        compute_delta_rank(j_base, tl_base), known_mask,
    )

    n_partners = len(prots)
    eps = args.epsilon
    rri_values = []

    for it in range(args.n_iter):
        noise_psi = noise_j = noise_s = None
        if args.scenario in ("psi_noise", "combined"):
            noise_psi = rng.uniform(1 - eps, 1 + eps, n_partners)
        if args.scenario in ("j_noise", "combined"):
            noise_j = rng.uniform(1 - eps, 1 + eps, n_partners)
        if args.scenario in ("s_noise", "combined"):
            noise_s = rng.uniform(1 - eps, 1 + eps, n_partners)
        if args.scenario == "conc_noise":
            noise_j = rng.uniform(1 - eps, 1 + eps, n_partners)

        _, tl_noisy, _ = score_all_pairs(
            anchor_row, prots, s_dict, args.species,
            noise_psi=noise_psi, noise_j=noise_j, noise_s=noise_s,
        )
        dr = compute_delta_rank(j_base, tl_noisy)
        rri_values.append(compute_rri(dr, known_mask))

    rri_arr = np.array(rri_values, dtype=float)
    valid = rri_arr[np.isfinite(rri_arr)]

    def _safe(fn, arr):
        return float(fn(arr)) if len(arr) > 0 else np.nan

    result = {
        "species": args.species,
        "anchor": args.anchor,
        "scenario": args.scenario,
        "epsilon": eps,
        "n_iter": args.n_iter,
        "baseline_rri": baseline_rri,
        "mean_rri": _safe(np.mean, valid),
        "std_rri": _safe(np.std, valid),
        "ci95_low": _safe(lambda a: np.percentile(a, 2.5), valid),
        "ci95_high": _safe(lambda a: np.percentile(a, 97.5), valid),
        "ci99_low": _safe(lambda a: np.percentile(a, 0.5), valid),
        "ci99_high": _safe(lambda a: np.percentile(a, 99.5), valid),
        "positive_ratio": _safe(lambda a: np.mean(a > 0), valid),
        "flip_probability": _safe(lambda a: np.mean(a <= 0), valid),
    }

    csv_path = out_dir / f"hpc2_{tag}.csv"
    to_csv_safe(pd.DataFrame([result]), csv_path)
    log.info(f"DONE   mean_rri={result['mean_rri']:.3f}  flip={result['flip_probability']:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--scenario", required=True,
                    choices=["psi_noise", "j_noise", "s_noise", "conc_noise", "combined"])
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        tag = f"{args.species}_{args.anchor}_{args.scenario}_eps{args.epsilon:.2f}"
        err_dir = Path(args.output_dir).parent / "logs" / "hpc2"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
