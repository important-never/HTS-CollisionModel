#!/usr/bin/env python3
"""
HPC-6: Leave-One-Out S-layer Robustness
========================================
ONE task = remove ONE S-layer kinetics entry and recompute RRI for 3 anchors.
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
    setup_task_logger, S_NEUTRAL, to_csv_safe,
)

ANCHORS = ["GTP", "Acetyl-CoA", "ATP"]


def get_kinetics_entries(s_dict, species):
    """Return deduplicated non-neutral S entries, matching prepare_data.py order."""
    raw = sorted(
        k for k, v in s_dict.items()
        if len(k) == 3 and k[2] == species and v != S_NEUTRAL and v > 0
    )
    seen = set()
    entries = []
    for key in raw:
        canonical = tuple(sorted([key[0], key[1]])) + (species,)
        if canonical in seen:
            continue
        seen.add(canonical)
        entries.append(key)
    return entries


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_rm{args.remove_idx:04d}"
    log_dir = out_dir.parent / "logs" / "hpc6"
    log = setup_task_logger("hpc6", log_dir, tag)

    master = load_master(args.data_dir)
    s_dict = load_s_values(args.data_dir)
    known = load_known_pairs(args.data_dir)

    entries = get_kinetics_entries(s_dict, args.species)
    if args.remove_idx >= len(entries):
        log.warning(f"SKIP  remove_idx {args.remove_idx} >= {len(entries)}")
        return

    removed_key = entries[args.remove_idx]
    removed_val = s_dict[removed_key]
    log.info(f"Removing {removed_key}  S={removed_val:.2e}")

    s_mod = dict(s_dict)
    s_mod[removed_key] = S_NEUTRAL
    rev_key = (removed_key[1], removed_key[0], removed_key[2])
    if rev_key in s_mod:
        s_mod[rev_key] = S_NEUTRAL

    sp_df = master[master["sp"] == args.species]
    prots = sp_df[sp_df["Category"] == "Protein"]
    mets = sp_df[sp_df["Category"] == "Metabolite"]
    known_set = known.get(args.species, set())

    results = []
    for anchor_name in ANCHORS:
        anchor_rows = mets[mets["Node_Name"] == anchor_name]
        if anchor_rows.empty:
            continue
        anchor_row = anchor_rows.iloc[0]

        known_mask = None
        original_rri = new_rri = np.nan

        for s_d, label in [(s_dict, "original"), (s_mod, "removed")]:
            j_scores, tl_scores, pnames = score_all_pairs(
                anchor_row, prots, s_d, args.species,
            )
            if known_mask is None:
                known_mask = np.array([
                    (anchor_name, pn) in known_set or (pn, anchor_name) in known_set
                    for pn in pnames
                ])
            dr = compute_delta_rank(j_scores, tl_scores)
            rri = compute_rri(dr, known_mask)
            if label == "original":
                original_rri = rri
            else:
                new_rri = rri

        delta = (
            new_rri - original_rri
            if np.isfinite(original_rri) and np.isfinite(new_rri)
            else np.nan
        )
        pct = (
            (delta / abs(original_rri) * 100)
            if np.isfinite(delta) and abs(original_rri) > 1e-9
            else np.nan
        )

        results.append({
            "species": args.species,
            "removed_pair": f"{removed_key[0]}|{removed_key[1]}",
            "removed_s_value": removed_val,
            "anchor": anchor_name,
            "original_rri": original_rri,
            "new_rri": new_rri,
            "delta_change": delta,
            "pct_change": pct,
            "critical_flag": abs(pct) > 30 if np.isfinite(pct) else False,
        })

    csv_path = out_dir / f"hpc6_{tag}.csv"
    to_csv_safe(pd.DataFrame(results), csv_path)
    log.info(f"DONE   {len(results)} rows → {csv_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--remove_idx", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        tag = f"{args.species}_rm{args.remove_idx:04d}"
        err_dir = Path(args.output_dir).parent / "logs" / "hpc6"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
