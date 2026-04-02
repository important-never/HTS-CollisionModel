#!/usr/bin/env python3
"""
Merge & Visualise HPC Results  (HPC-architect grade)
=====================================================
Merges per-task CSVs into summary files.
All figures → 300 dpi PDF + SVG (publication-ready).

Usage:
  python scripts/merge_results.py --results_dir results --output_dir final_output
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DPI = 300
STAR_MOLS = {"GTP", "Acetyl-CoA", "ATP"}

def _to_csv(df, path):
    try:
        df.to_csv(path, index=False, line_terminator="\n")
    except TypeError:
        df.to_csv(path, index=False, lineterminator="\n")


def _glob_csvs(d: Path, prefix: str):
    if not d.exists():
        return []
    return sorted(d.glob(f"{prefix}_*.csv"))


def _read_all(files):
    frames = []
    for f in files:
        raw = f.read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        import io
        frames.append(pd.read_csv(io.StringIO(text)))
    return pd.concat(frames, ignore_index=True)


def _savefig(fig, stem: str, out: Path):
    """Save figure as PDF and SVG at 300 dpi."""
    for ext in ("pdf", "svg"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figures saved: {stem}.pdf / .svg")


# ── HPC-1 ──────────────────────────────────────────────────────────────

def merge_hpc1(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc1", "hpc1")
    if not files:
        print("[HPC-1] No result files found"); return
    df = _read_all(files)
    df.sort_values(["species", "mean_delta_rank"], ascending=[True, False], inplace=True)
    csv_out = out / "hpc1_metabolite_atlas_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-1] Merged {len(files)} files → {csv_out.name} ({len(df)} rows)")

    # Fig 1 — RRI waterfall per species
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    for ax, sp in zip(axes, ["E.coli", "Mammalian", "Yeast"]):
        sub = df[df["species"] == sp].sort_values("mean_delta_rank", ascending=False)
        colors = ["#c0392b" if a in STAR_MOLS else "#2980b9" for a in sub["anchor"]]
        ax.barh(range(len(sub)), sub["mean_delta_rank"], color=colors, edgecolor="none")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["anchor"], fontsize=6)
        ax.set_xlabel("Rank Rescue Index")
        ax.set_title(sp, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle("HPC-1: Genome-wide Rank Rescue Atlas", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "hpc1_metabolite_atlas", out)

    # Fig 2 — RRI vs connectivity (Hub Curse)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for sp, mk in [("E.coli", "o"), ("Mammalian", "s"), ("Yeast", "^")]:
        sub = df[df["species"] == sp]
        ax2.scatter(sub["n_known"], sub["mean_delta_rank"], alpha=0.55,
                    label=sp, marker=mk, s=40, edgecolors="k", linewidths=0.3)
        for _, row in sub[sub["anchor"].isin(STAR_MOLS)].iterrows():
            ax2.annotate(row["anchor"],
                         (row["n_known"], row["mean_delta_rank"]),
                         fontsize=8, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("Known partners (connectivity)")
    ax2.set_ylabel("Rank Rescue Index")
    ax2.set_title("RRI vs Connectivity — Hub Ranking Curse", fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    _savefig(fig2, "hpc1_rri_vs_connectivity", out)


# ── HPC-1P (Protein-Anchor Flipped) ───────────────────────────────────────

def merge_hpc1p(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc1p", "hpc1p")
    if not files:
        print("[HPC-1P] No result files found"); return
    df = _read_all(files)
    df.sort_values(["species", "mean_delta_rank"], ascending=[True, False], inplace=True)
    csv_out = out / "hpc1p_protein_atlas_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-1P] Merged {len(files)} files -> {csv_out.name} ({len(df)} rows)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    for ax, sp in zip(axes, ["E.coli", "Mammalian", "Yeast"]):
        sub = df[df["species"] == sp].sort_values("mean_delta_rank", ascending=False)
        ax.barh(range(len(sub)), sub["mean_delta_rank"], color="#2980b9", edgecolor="none")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["anchor"], fontsize=6)
        ax.set_xlabel("Rank Rescue Index")
        ax.set_title(sp, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle("HPC-1P: Protein-Anchor Rank Rescue Atlas", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "hpc1p_protein_atlas", out)


# ── HPC-2 ──────────────────────────────────────────────────────────────

def merge_hpc2(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc2", "hpc2")
    if not files:
        print("[HPC-2] No result files found"); return
    df = _read_all(files)
    csv_out = out / "hpc2_monte_carlo_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-2] Merged {len(files)} files → {csv_out.name}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, anchor in zip(axes, ["GTP", "Acetyl-CoA", "ATP"]):
        sub = df[(df["anchor"] == anchor) & (df["scenario"] == "combined")]
        for sp, ls in [("E.coli", "-"), ("Mammalian", "--"), ("Yeast", ":")]:
            sp_sub = sub[sub["species"] == sp].sort_values("epsilon")
            if sp_sub.empty:
                continue
            ax.plot(sp_sub["epsilon"], sp_sub["positive_ratio"],
                    label=sp, linestyle=ls, marker="o", markersize=4)
        ax.set_xlabel("Noise amplitude (ε)")
        ax.set_ylabel("Positive ratio")
        ax.set_title(anchor, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.5, label="95 % threshold")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("HPC-2: Flip Threshold Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "hpc2_flip_threshold", out)


# ── HPC-3 ──────────────────────────────────────────────────────────────

def merge_hpc3(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc3", "hpc3")
    if not files:
        print("[HPC-3] No result files found"); return
    df = _read_all(files)
    csv_out = out / "hpc3_phase_space_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-3] Merged {len(files)} files → {csv_out.name}")

    for anchor in ["GTP", "Acetyl-CoA", "ATP"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, sp in zip(axes, ["E.coli", "Mammalian", "Yeast"]):
            sub = df[(df["anchor"] == anchor) & (df["species"] == sp)]
            mid_val = sub["psi_midpoint"].median() if len(sub) > 0 else 40000
            sub_2d = sub[abs(sub["psi_midpoint"] - mid_val) < 1]
            if sub_2d.empty:
                sub_2d = sub
            if sub_2d.empty:
                continue
            pivot = sub_2d.pivot_table(
                index="eta_factor", columns="T", values="mean_delta_rank",
            )
            if pivot.empty:
                continue
            im = ax.imshow(
                pivot.values, aspect="auto", origin="lower",
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max()],
                cmap="RdYlBu_r",
            )
            ax.plot(310.15, 3.0, "k*", markersize=14, label="Biology (37C, eta x3)")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Crowding factor")
            ax.set_title(sp, fontweight="bold")
            plt.colorbar(im, ax=ax, label="RRI", shrink=0.85)
            ax.legend(fontsize=7)
        safe = anchor.replace("-", "").replace(" ", "")
        fig.suptitle(f"HPC-3: Phase Diagram — {anchor}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _savefig(fig, f"hpc3_phase_{safe}", out)


# ── HPC-4 ──────────────────────────────────────────────────────────────

def merge_hpc4(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc4", "hpc4")
    if not files:
        print("[HPC-4] No result files found"); return
    df = _read_all(files)
    df.sort_values(["species", "growth_impact"], ascending=[True, False], inplace=True)
    csv_out = out / "hpc4_gem_cascade_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-4] Merged {len(files)} files → {csv_out.name}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, sp in zip(axes, ["E.coli", "Yeast", "Mammalian"]):
        sub = df[df["species"] == sp].nlargest(30, "growth_impact")
        colors = []
        for _, row in sub.iterrows():
            mid = str(row["met_id"]).lower()
            if "gtp" in mid:   colors.append("#c0392b")
            elif "accoa" in mid: colors.append("#e67e22")
            elif "atp" in mid:  colors.append("#27ae60")
            else:               colors.append("#2980b9")
        ax.barh(range(len(sub)), sub["growth_impact"], color=colors, edgecolor="none")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([str(r["met_id"]) for _, r in sub.iterrows()], fontsize=6)
        ax.set_xlabel("Growth Impact")
        ax.set_title(f"{sp} — Top 30", fontweight="bold")
        ax.invert_yaxis()
    fig.suptitle("HPC-4: GEM Control Hierarchy", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "hpc4_gem_control_hierarchy", out)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for sp, mk in [("E.coli", "o"), ("Yeast", "s"), ("Mammalian", "^")]:
        sub = df[df["species"] == sp]
        ax2.scatter(sub["n_reactions"], sub["growth_impact"], alpha=0.4,
                    label=sp, marker=mk, s=20, edgecolors="k", linewidths=0.2)
    ax2.set_xlabel("Reactions (connectivity)")
    ax2.set_ylabel("Growth Impact")
    ax2.set_title("Connectivity vs Growth Impact (TDR Test)", fontweight="bold")
    ax2.legend(frameon=True)
    _savefig(fig2, "hpc4_connectivity_vs_impact", out)


# ── HPC-6 ──────────────────────────────────────────────────────────────

def merge_hpc6(results_dir: Path, out: Path):
    files = _glob_csvs(results_dir / "hpc6", "hpc6")
    if not files:
        print("[HPC-6] No result files found"); return
    df = _read_all(files)
    csv_out = out / "hpc6_loo_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-6] Merged {len(files)} files → {csv_out.name}")

    n_crit = int(df["critical_flag"].sum()) if "critical_flag" in df.columns else 0
    print(f"  Critical data points (>30 % RRI change): {n_crit}")

    fig, ax = plt.subplots(figsize=(12, 6))
    offset = 0
    for anchor, color in [("GTP", "#c0392b"), ("Acetyl-CoA", "#e67e22"), ("ATP", "#27ae60")]:
        sub = df[df["anchor"] == anchor].sort_values("pct_change")
        if sub.empty:
            continue
        y_pos = range(offset, offset + len(sub))
        ax.barh(list(y_pos), sub["pct_change"].values, color=color, alpha=0.7, label=anchor)
        offset += len(sub)
    ax.set_xlabel("% change in RRI when data point removed")
    ax.set_title("HPC-6: Leave-One-Out Sensitivity", fontweight="bold")
    ax.axvline(-30, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(30,  color="red", ls="--", lw=0.8, alpha=0.5)
    ax.legend(frameon=True)
    _savefig(fig, "hpc6_loo_waterfall", out)


# ── Main ───────────────────────────────────────────────────────────────

def merge_hpc7(results_dir: Path, out: Path):
    """Merge S-shuffle causality test results."""
    files = _glob_csvs(results_dir / "hpc7", "hpc7")
    if not files:
        print("[HPC-7] No result files found"); return
    df = _read_all(files)
    csv_out = out / "hpc7_s_shuffle_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-7] Merged {len(files)} files → {csv_out.name} ({len(df)} rows)")

    # Summary: real vs shuffled RRI per species×anchor
    summary = df.groupby(["species", "anchor"]).agg(
        n_seeds=("seed", "count"),
        n_known=("n_known", "first"),
        rri_real_mean=("rri_real", "mean"),
        rri_shuffled_mean=("rri_shuffled", "mean"),
        rri_shuffled_std=("rri_shuffled", "std"),
        rri_delta_mean=("rri_delta", "mean"),
    ).reset_index()
    _to_csv(summary, out / "hpc7_s_shuffle_summary.csv")
    print(f"  Summary: {len(summary)} species×anchor combos")

    # Wilcoxon test: is rri_delta > 0 across all seeds?
    from scipy.stats import wilcoxon
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        deltas = df[df["species"] == sp]["rri_delta"].dropna().values
        if len(deltas) >= 5:
            try:
                stat, p = wilcoxon(deltas, alternative="greater")
                print(f"  {sp} Wilcoxon: delta_mean={deltas.mean():.2f}, p={p:.2e}, n={len(deltas)}")
            except Exception:
                print(f"  {sp} Wilcoxon: failed (n={len(deltas)})")

    # Fig — boxplot real vs shuffled
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sp in zip(axes, ["E.coli", "Mammalian", "Yeast"]):
        sub = df[df["species"] == sp]
        data = [sub["rri_real"].dropna(), sub["rri_shuffled"].dropna()]
        bp = ax.boxplot(data, labels=["Real S", "Shuffled S"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_ylabel("RRI")
        ax.set_title(sp, fontweight="bold")
    fig.suptitle("HPC-7: S-layer Shuffle Causality Test", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "hpc7_s_shuffle_boxplot", out)


def merge_hpc8(results_dir: Path, out: Path):
    """Merge log-normal noise MC results."""
    files = _glob_csvs(results_dir / "hpc8", "hpc8")
    if not files:
        print("[HPC-8] No result files found"); return
    df = _read_all(files)
    csv_out = out / "hpc8_lognormal_mc_combined.csv"
    _to_csv(df, csv_out)
    print(f"[HPC-8] Merged {len(files)} files → {csv_out.name} ({len(df)} rows)")

    # Summary: flip probability vs sigma
    summary = df.groupby(["species", "anchor", "sigma"]).agg(
        n_seeds=("seed", "count"),
        n_known=("n_known", "first"),
        baseline_rri=("baseline_rri", "mean"),
        mean_rri=("mean_rri", "mean"),
        mean_flip=("flip_probability", "mean"),
        mean_pos_ratio=("mean_pos_ratio", "mean"),
    ).reset_index()
    _to_csv(summary, out / "hpc8_lognormal_mc_summary.csv")
    print(f"  Summary: {len(summary)} combos")

    # Fig — flip probability vs sigma (breaking point curve)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, anchor in zip(axes, ["GTP", "Acetyl-CoA", "ATP"]):
        for sp, color in [("E.coli", "#2980b9"), ("Mammalian", "#e74c3c"), ("Yeast", "#27ae60")]:
            sub = summary[(summary["anchor"] == anchor) & (summary["species"] == sp)]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("sigma")
            ax.plot(sub["sigma"], sub["mean_flip"], "o-", color=color, label=sp, markersize=5)
        ax.set_xlabel("σ (log-normal noise)")
        ax.set_ylabel("Flip probability")
        ax.set_title(f"★ {anchor}", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    fig.suptitle("HPC-8: Breaking Point Curve (Log-Normal Noise)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "hpc8_breaking_point_curve", out)


def compute_filtered_stats(results_dir: Path, out: Path):
    """Compute n_known>=3 filtered statistics + global Wilcoxon for HPC-1."""
    files = _glob_csvs(results_dir / "hpc1", "hpc1")
    if not files:
        print("[Filtered Stats] No HPC-1 files"); return
    df = _read_all(files)

    from scipy.stats import wilcoxon

    print("\n[Filtered Stats] n_known≥3 analysis:")
    rows = []
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        sub_all = df[df["species"] == sp]
        sub_filt = sub_all[sub_all["n_known"] >= 3]
        n_all = len(sub_all)
        n_filt = len(sub_filt)

        # Significance rate
        sig_all = (sub_all["permutation_p"] < 0.05).sum() if "permutation_p" in sub_all.columns else 0
        sig_filt = (sub_filt["permutation_p"] < 0.05).sum() if "permutation_p" in sub_filt.columns else 0

        # Global Wilcoxon on RRI
        rri_all = sub_all["mean_delta_rank"].dropna().values
        rri_filt = sub_filt["mean_delta_rank"].dropna().values

        w_p_all = w_p_filt = np.nan
        if len(rri_all) >= 5:
            try:
                _, w_p_all = wilcoxon(rri_all, alternative="greater")
            except Exception:
                pass
        if len(rri_filt) >= 5:
            try:
                _, w_p_filt = wilcoxon(rri_filt, alternative="greater")
            except Exception:
                pass

        print(f"  {sp}: all={n_all} (sig={sig_all}/{n_all}={100*sig_all/max(n_all,1):.1f}%), "
              f"n≥3={n_filt} (sig={sig_filt}/{n_filt}={100*sig_filt/max(n_filt,1):.1f}%)")
        print(f"    Wilcoxon(all): p={w_p_all:.2e}, Wilcoxon(n≥3): p={w_p_filt:.2e}")

        rows.append({
            "species": sp,
            "n_all": n_all, "n_filtered": n_filt,
            "sig_all": sig_all, "sig_filtered": sig_filt,
            "sig_rate_all": sig_all / max(n_all, 1),
            "sig_rate_filtered": sig_filt / max(n_filt, 1),
            "wilcoxon_p_all": w_p_all,
            "wilcoxon_p_filtered": w_p_filt,
            "mean_rri_all": float(rri_all.mean()) if len(rri_all) > 0 else np.nan,
            "mean_rri_filtered": float(rri_filt.mean()) if len(rri_filt) > 0 else np.nan,
        })

    _to_csv(pd.DataFrame(rows), out / "hpc1_filtered_stats.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HPC Results Merger & Visualiser  (300 dpi PDF/SVG)")
    print("=" * 60)

    merge_hpc1(args.results_dir, out)
    merge_hpc1p(args.results_dir, out)
    merge_hpc2(args.results_dir, out)
    merge_hpc3(args.results_dir, out)
    merge_hpc4(args.results_dir, out)
    merge_hpc6(args.results_dir, out)
    merge_hpc7(args.results_dir, out)
    merge_hpc8(args.results_dir, out)
    compute_filtered_stats(args.results_dir, out)

    print(f"\n{'='*60}")
    print(f"  All merging complete!  Output → {out.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
