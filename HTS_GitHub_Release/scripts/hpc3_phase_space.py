#!/usr/bin/env python3
"""
HPC-3: Physical Parameter Phase Space Mapping
==============================================
ONE task = ONE (T, eta_factor, psi_midpoint) grid point.
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
    setup_task_logger, ETA_WATER, to_csv_safe,
)

ANCHORS = ["GTP", "Acetyl-CoA", "ATP"]
SPECIES_LIST = ["E.coli", "Mammalian", "Yeast"]


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"T{args.T:.1f}_eta{args.eta_factor:.1f}_psi{args.psi_midpoint:.0f}"
    log_dir = out_dir.parent / "logs" / "hpc3"
    log = setup_task_logger("hpc3", log_dir, tag)

    log.info(f"START  T={args.T}  eta_factor={args.eta_factor}  psi_mid={args.psi_midpoint}")

    master = load_master(args.data_dir)
    s_dict = load_s_values(args.data_dir)
    known = load_known_pairs(args.data_dir)
    eta = ETA_WATER * args.eta_factor

    results = []
    for sp in SPECIES_LIST:
        sp_df = master[master["sp"] == sp]
        prots = sp_df[sp_df["Category"] == "Protein"]
        mets = sp_df[sp_df["Category"] == "Metabolite"]
        known_set = known.get(sp, set())

        for anchor_name in ANCHORS:
            anchor_rows = mets[mets["Node_Name"] == anchor_name]
            if anchor_rows.empty:
                continue
            anchor_row = anchor_rows.iloc[0]

            j_scores, tl_scores, partner_names = score_all_pairs(
                anchor_row, prots, s_dict, sp,
                T=args.T, eta=eta, psi_midpoint=args.psi_midpoint,
            )
            dr = compute_delta_rank(j_scores, tl_scores)
            known_mask = np.array([
                (anchor_name, pn) in known_set or (pn, anchor_name) in known_set
                for pn in partner_names
            ])
            rri = compute_rri(dr, known_mask)
            pos_ratio = (
                float(np.mean(dr[known_mask] > 0))
                if known_mask.sum() > 0 else np.nan
            )
            top_idx = int(np.argmax(dr))

            results.append({
                "T": args.T,
                "eta_factor": args.eta_factor,
                "psi_midpoint": args.psi_midpoint,
                "species": sp,
                "anchor": anchor_name,
                "mean_delta_rank": rri,
                "positive_ratio": pos_ratio,
                "n_known": int(known_mask.sum()),
                "top1_partner": partner_names[top_idx],
                "top1_delta": float(dr[top_idx]),
            })

    csv_path = out_dir / f"hpc3_{tag}.csv"
    to_csv_safe(pd.DataFrame(results), csv_path)
    log.info(f"DONE   {len(results)} rows → {csv_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--eta_factor", type=float, required=True)
    ap.add_argument("--psi_midpoint", type=float, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        tag = f"T{args.T:.1f}_eta{args.eta_factor:.1f}_psi{args.psi_midpoint:.0f}"
        err_dir = Path(args.output_dir).parent / "logs" / "hpc3"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
