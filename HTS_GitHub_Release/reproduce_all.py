#!/usr/bin/env python3
"""
HTS Model — One-Click Reproduction Script
==========================================

Reproduces the core computational results cited in the paper:
  "Toward a networked Central Dogma: physical constraints predict
   molecular encounter specificity across three kingdoms"

This script covers single-machine reproducible analyses:
  1. Three-layer model scoring (Psi x J x S) for all species
  2. S-shuffle causality test (paper Table 1 / HPC-7 equivalent)
  3. Cross-species RRI correlation audit (paper Section 2.4)
  4. Log-normal breaking point (paper Table 1 / HPC-8 equivalent)

For full HPC experiments (25,727 tasks on 64-core cluster), contact
the authors. HPC scripts are provided in scripts/ for reference.

Usage:
    python reproduce_all.py              # Full reproduction (~20-40 min)
    python reproduce_all.py --quick      # Quick smoke test (~2 min)
    python reproduce_all.py --audit      # Cross-species audit only (~5 min)

Requirements:
    pip install -r requirements.txt
"""
import sys, os, time, argparse, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from hpc_common import (
    load_master, load_s_values, load_known_pairs,
    J_ij, delta_comp, S_NEUTRAL,
    score_all_pairs, compute_delta_rank, compute_rri,
    permutation_test_rri, to_csv_safe,
)

DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results"
REPORT   = ROOT / "REPRODUCTION_REPORT.txt"

SPECIES = ["E.coli", "Mammalian", "Yeast"]
HUBS    = ["GTP", "Acetyl-CoA", "ATP"]

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Logger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.start = time.time()

    def log(self, msg=""):
        elapsed = time.time() - self.start
        line = f"[{elapsed:7.1f}s] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def build_known_mask(partner_names, known_pairs, anchor_name):
    """Build boolean mask: True if (anchor, partner) is a known pair."""
    kp_set = set()
    for pair in known_pairs:
        a, b = pair[0], pair[1]
        if a == anchor_name:
            kp_set.add(b)
        elif b == anchor_name:
            kp_set.add(a)
    return np.array([pn in kp_set for pn in partner_names])


def load_data(log):
    log.log("Loading master data ...")
    master = load_master(DATA_DIR)
    s_dict = load_s_values(DATA_DIR)
    kp     = load_known_pairs(DATA_DIR)
    log.log(f"  V41_Master: {len(master)} nodes")
    for sp in SPECIES:
        sp_df = master[master["sp"] == sp]
        n_met = len(sp_df[sp_df["Category"] == "Metabolite"])
        n_pro = len(sp_df[sp_df["Category"] == "Protein"])
        log.log(f"  {sp}: {n_met} metabolites + {n_pro} proteins = {len(sp_df)} total")
    n_s = sum(1 for v in s_dict.values() if v != S_NEUTRAL)
    log.log(f"  S-values: {n_s:,} non-neutral entries")
    for sp in SPECIES:
        log.log(f"  Known pairs ({sp}): {len(kp.get(sp, []))}")
    return master, s_dict, kp


# ===== EXP 1: Metabolite Atlas (single-machine version of HPC-1) =====

def run_metabolite_atlas(master, s_dict, kp, log, quick=False):
    log.log("")
    log.log("=" * 60)
    log.log("EXPERIMENT 1: Metabolite Atlas (HPC-1 equivalent)")
    log.log("=" * 60)
    rows = []
    for sp in SPECIES:
        sp_df = master[master["sp"] == sp].copy()
        mets = sp_df[sp_df["Category"] == "Metabolite"]
        anchors = mets["Node_Name"].tolist()
        if quick:
            anchors = [a for a in anchors if a in HUBS]

        log.log(f"  {sp}: scoring {len(anchors)} metabolite anchors ...")
        for anc_name in anchors:
            anc_row = sp_df[sp_df["Node_Name"] == anc_name].iloc[0]
            partner_df = sp_df[sp_df["Node_Name"] != anc_name].reset_index(drop=True)

            j_scores, tl_scores, p_names = score_all_pairs(
                anc_row, partner_df, s_dict, sp)
            dr = compute_delta_rank(j_scores, tl_scores)
            known_mask = build_known_mask(p_names, kp.get(sp, []), anc_name)

            n_known = int(known_mask.sum())
            if n_known == 0:
                continue
            rri = compute_rri(dr, known_mask)
            pos = int(np.sum(dr[known_mask] > 0))
            pos_ratio = pos / n_known
            p_val = permutation_test_rri(dr, known_mask,
                                         n_perm=200 if quick else 1000)

            rows.append(dict(species=sp, anchor=anc_name, n_known=n_known,
                             mean_rri=round(rri, 2),
                             pos_ratio=round(pos_ratio, 3),
                             p_value=round(p_val, 4)))

    df = pd.DataFrame(rows)
    to_csv_safe(df, OUT_DIR / "exp1_metabolite_atlas.csv")
    log.log(f"  Saved {len(df)} rows -> exp1_metabolite_atlas.csv")

    for sp in SPECIES:
        sub = df[df["species"] == sp]
        sig = sub[sub["p_value"] < 0.05]
        n3 = sub[sub["n_known"] >= 3]
        sig3 = n3[n3["p_value"] < 0.05]
        log.log(f"  {sp}: {len(sig)}/{len(sub)} significant (all), "
                f"{len(sig3)}/{len(n3)} significant (n_known>=3)")

    for sp in SPECIES:
        sp_rows = df[(df["species"] == sp) & (df["n_known"] >= 3)]
        if len(sp_rows) >= 3:
            stat, p = wilcoxon(sp_rows["mean_rri"], alternative="greater")
            log.log(f"  {sp} global Wilcoxon (n_known>=3): p = {p:.2e}")

    # Hub detail
    log.log("  --- Hub molecules ---")
    for _, r in df[df["anchor"].isin(HUBS)].iterrows():
        log.log(f"    {r['species']}/{r['anchor']}: RRI={r['mean_rri']:.1f}  "
                f"n_known={r['n_known']}  p={r['p_value']:.4f}")
    return df


# ===== EXP 2: S-Shuffle Causality (HPC-7 equivalent) =====

def run_s_shuffle(master, s_dict, kp, log, quick=False):
    log.log("")
    log.log("=" * 60)
    log.log("EXPERIMENT 2: S-Shuffle Causality Test (HPC-7 equivalent)")
    log.log("=" * 60)
    n_seeds = 10 if quick else 100
    rows = []

    for sp in SPECIES:
        sp_df = master[master["sp"] == sp].copy()
        sp_keys = [(a, b, s) for (a, b, s) in s_dict if s == sp]
        s_vals_arr = np.array([s_dict[k] for k in sp_keys])

        for anc_name in HUBS:
            anc_match = sp_df[sp_df["Node_Name"] == anc_name]
            if anc_match.empty:
                continue
            anc_row = anc_match.iloc[0]
            partner_df = sp_df[sp_df["Node_Name"] != anc_name].reset_index(drop=True)

            j_real, tl_real, p_names = score_all_pairs(
                anc_row, partner_df, s_dict, sp)
            dr_real = compute_delta_rank(j_real, tl_real)
            known_mask = build_known_mask(p_names, kp.get(sp, []), anc_name)
            n_known = int(known_mask.sum())
            if n_known == 0:
                continue
            rri_real = compute_rri(dr_real, known_mask)

            shuffle_rris = []
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed)
                shuffled_vals = rng.permutation(s_vals_arr)
                s_shuffled = dict(s_dict)
                for i, k in enumerate(sp_keys):
                    s_shuffled[k] = shuffled_vals[i]

                _, tl_sh, _ = score_all_pairs(
                    anc_row, partner_df, s_shuffled, sp)
                dr_sh = compute_delta_rank(j_real, tl_sh)
                shuffle_rris.append(compute_rri(dr_sh, known_mask))

            mean_sh = np.mean(shuffle_rris)
            rows.append(dict(species=sp, anchor=anc_name, n_known=n_known,
                             rri_real=round(rri_real, 2),
                             rri_shuffled=round(mean_sh, 2),
                             delta=round(rri_real - mean_sh, 2),
                             n_seeds=n_seeds))
            log.log(f"  {sp}/{anc_name}: real={rri_real:.1f}  "
                    f"shuffled={mean_sh:.1f}  delta={rri_real - mean_sh:+.1f}")

    df = pd.DataFrame(rows)
    to_csv_safe(df, OUT_DIR / "exp2_s_shuffle.csv")
    log.log(f"  Saved -> exp2_s_shuffle.csv")

    for sp in SPECIES:
        sub = df[df["species"] == sp]
        if len(sub) >= 3:
            stat, p = wilcoxon(sub["delta"], alternative="greater")
            log.log(f"  {sp} Wilcoxon (delta>0): p = {p:.2e}")
    return df


# ===== EXP 3: Cross-Species RRI Audit =====

def run_cross_species_audit(master, s_dict, log):
    log.log("")
    log.log("=" * 60)
    log.log("EXPERIMENT 3: Cross-Species RRI Audit (Section 2.4)")
    log.log("=" * 60)

    all_data = {}
    for sp in SPECIES:
        sp_df = master[master["sp"] == sp]
        mets = sp_df[sp_df["Category"] == "Metabolite"]
        all_nodes = sp_df.to_dict("records")
        result = {}

        for _, anc_row in mets.iterrows():
            anc_name = anc_row["Node_Name"]
            anc_mw   = float(anc_row["MW_Da"])
            anc_conc = float(anc_row["Conc_uM"]) if pd.notna(anc_row["Conc_uM"]) and anc_row["Conc_uM"] > 0 else 1.0
            anc_comp = str(anc_row["Compartment"])

            j_list, tl_list, names, cats = [], [], [], []
            for node in all_nodes:
                pn = node["Node_Name"]
                if pn == anc_name:
                    continue
                p_mw   = float(node["MW_Da"])   if pd.notna(node["MW_Da"])   and node["MW_Da"] > 0   else 50000
                p_conc = float(node["Conc_uM"]) if pd.notna(node["Conc_uM"]) and node["Conc_uM"] > 0 else 1.0
                p_comp = str(node["Compartment"])
                psi = delta_comp(anc_comp, p_comp, anc_mw, p_mw)
                j   = J_ij(anc_mw, p_mw, anc_conc, p_conc)
                s   = s_dict.get((anc_name, pn, sp),
                      s_dict.get((pn, anc_name, sp), S_NEUTRAL))
                j_list.append(psi * j)
                tl_list.append(psi * j * s)
                names.append(pn)
                cats.append(node["Category"])

            j_arr  = np.array(j_list)
            tl_arr = np.array(tl_list)
            rk_j  = rankdata(-j_arr,  method="min")
            rk_tl = rankdata(-tl_arr, method="min")
            dr = rk_j - rk_tl
            for i, pn in enumerate(names):
                result[(anc_name, pn)] = dict(
                    rank_j=rk_j[i], rank_tl=rk_tl[i],
                    delta_rank=dr[i], cat=cats[i])

        all_data[sp] = result
        log.log(f"  {sp}: {len(result):,} (anchor, partner) pairs computed")

    pairs_list = [("E.coli", "Mammalian"), ("E.coli", "Yeast"), ("Mammalian", "Yeast")]
    detail_rows, summary_rows = [], []

    for s1, s2 in pairs_list:
        d1, d2 = all_data[s1], all_data[s2]
        shared = sorted(k for k in set(d1) & set(d2) if d1[k]["cat"] == "Metabolite")
        n = len(shared)
        dr1 = np.array([d1[k]["delta_rank"] for k in shared])
        dr2 = np.array([d2[k]["delta_rank"] for k in shared])
        j1  = np.array([d1[k]["rank_j"] for k in shared])
        j2  = np.array([d2[k]["rank_j"] for k in shared])
        tl1 = np.array([d1[k]["rank_tl"] for k in shared])
        tl2 = np.array([d2[k]["rank_tl"] for k in shared])

        rho_rri, _ = spearmanr(dr1, dr2)
        rho_j, _   = spearmanr(j1, j2)
        rho_tl, _  = spearmanr(tl1, tl2)
        same = int(np.sum(np.sign(dr1) == np.sign(dr2)))

        summary_rows.append(dict(
            sp1=s1, sp2=s2, n=n,
            rho_RRI=round(rho_rri, 4), rho_J=round(rho_j, 4),
            rho_full=round(rho_tl, 4), delta_rho=round(rho_tl - rho_j, 4),
            same_sign_frac=round(same / n, 4)))

        for i, k in enumerate(shared):
            detail_rows.append(dict(
                sp1=s1, sp2=s2, anchor=k[0], partner=k[1],
                dr_sp1=dr1[i], dr_sp2=dr2[i]))

        log.log(f"  {s1} vs {s2} (n={n}): RRI rho={rho_rri:.4f}  "
                f"J rho={rho_j:.4f}  Full rho={rho_tl:.4f}  "
                f"same-sign={same/n:.1%}")

    to_csv_safe(pd.DataFrame(detail_rows), OUT_DIR / "exp3_cross_species_detail.csv")
    to_csv_safe(pd.DataFrame(summary_rows), OUT_DIR / "exp3_cross_species_summary.csv")
    log.log(f"  Saved -> exp3_cross_species_detail.csv ({len(detail_rows)} rows)")
    log.log(f"  Saved -> exp3_cross_species_summary.csv")

    log.log("  Per-hub pairwise concordance:")
    detail_df = pd.DataFrame(detail_rows)
    for hub in HUBS:
        sub = detail_df[detail_df["anchor"] == hub]
        if sub.empty:
            continue
        same_t, n_t = 0, 0
        for _, grp in sub.groupby(["sp1", "sp2"]):
            s = np.sum(np.sign(grp["dr_sp1"].values) == np.sign(grp["dr_sp2"].values))
            same_t += s
            n_t += len(grp)
        log.log(f"    {hub}: {same_t}/{n_t} = {same_t/n_t:.1%}")


# ===== EXP 4: Breaking Point (HPC-8 equivalent) =====

def run_breaking_point(master, s_dict, kp, log, quick=False):
    log.log("")
    log.log("=" * 60)
    log.log("EXPERIMENT 4: Log-Normal Breaking Point (HPC-8 equivalent)")
    log.log("=" * 60)
    sigmas = [0.5, 1.0, 2.0, 3.0] if quick else [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    n_mc = 50 if quick else 200
    rows = []

    for sp in SPECIES:
        sp_df = master[master["sp"] == sp].copy()
        sp_keys = [(a, b, s) for (a, b, s) in s_dict if s == sp]

        for anc_name in HUBS:
            anc_match = sp_df[sp_df["Node_Name"] == anc_name]
            if anc_match.empty:
                continue
            anc_row = anc_match.iloc[0]
            partner_df = sp_df[sp_df["Node_Name"] != anc_name].reset_index(drop=True)

            j_real, tl_real, p_names = score_all_pairs(
                anc_row, partner_df, s_dict, sp)
            dr_real = compute_delta_rank(j_real, tl_real)
            known_mask = build_known_mask(p_names, kp.get(sp, []), anc_name)
            n_known = int(known_mask.sum())
            if n_known == 0:
                continue
            real_rri = compute_rri(dr_real, known_mask)
            real_sign = 1 if real_rri > 0 else -1

            for sigma in sigmas:
                flips = 0
                for seed in range(n_mc):
                    rng = np.random.RandomState(seed * 1000 + int(sigma * 10))
                    s_noisy = dict(s_dict)
                    for k in sp_keys:
                        noise = 10.0 ** rng.normal(0, sigma)
                        s_noisy[k] = s_dict[k] * noise

                    _, tl_n, _ = score_all_pairs(
                        anc_row, partner_df, s_noisy, sp)
                    dr_n = compute_delta_rank(j_real, tl_n)
                    noisy_rri = compute_rri(dr_n, known_mask)
                    if (1 if noisy_rri > 0 else -1) != real_sign:
                        flips += 1

                rows.append(dict(species=sp, anchor=anc_name, sigma=sigma,
                                 n_mc=n_mc, flip_prob=round(flips / n_mc, 4)))

            line_parts = [f"s={r['sigma']}:{r['flip_prob']:.3f}"
                          for r in rows if r["species"] == sp and r["anchor"] == anc_name]
            log.log(f"  {sp}/{anc_name}: " + "  ".join(line_parts))

    df = pd.DataFrame(rows)
    to_csv_safe(df, OUT_DIR / "exp4_breaking_point.csv")
    log.log(f"  Saved -> exp4_breaking_point.csv")
    return df


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(
        description="HTS Model: One-Click Reproduction")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (~2 min)")
    parser.add_argument("--audit", action="store_true",
                        help="Cross-species audit only (~5 min)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    log = Logger(REPORT)
    mode = "quick" if args.quick else ("audit" if args.audit else "full")
    log.log("HTS Model Reproduction Script")
    log.log(f"Mode: {mode}")
    log.log(f"Python: {sys.version.split()[0]}")
    log.log(f"Working dir: {ROOT}")
    log.log("")

    master, s_dict, kp = load_data(log)

    if args.audit:
        run_cross_species_audit(master, s_dict, log)
        log.log("\nDone.")
        log.close()
        return

    run_metabolite_atlas(master, s_dict, kp, log, quick=args.quick)
    run_s_shuffle(master, s_dict, kp, log, quick=args.quick)
    run_cross_species_audit(master, s_dict, log)
    run_breaking_point(master, s_dict, kp, log, quick=args.quick)

    log.log("")
    log.log("=" * 60)
    log.log("ALL EXPERIMENTS COMPLETE")
    log.log(f"Results:  {OUT_DIR}/")
    log.log(f"Report:   {REPORT}")
    log.log("=" * 60)
    log.log("")
    log.log("NOTE: Full HPC experiments (25,727 tasks) require a compute")
    log.log("cluster. Contact the authors for the raw HPC output data.")
    log.close()
    print(f"\nReport saved to: {REPORT}")


if __name__ == "__main__":
    main()
