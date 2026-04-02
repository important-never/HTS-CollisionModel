#!/usr/bin/env python3
"""
HPC Common Library — shared infrastructure for all HPC experiments.

Architecture principles:
  - pathlib for ALL path operations (cross-platform)
  - CRLF-safe CSV reader (Windows → Linux transfer)
  - Structured logging to independent files per task
  - Error-safe wrappers for parallel-job resilience
"""
import io
import logging
import math
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ── Physical Constants (identical to three_layer_model.py) ──────────────
K_B = 1.380649e-23
T_DEFAULT = 310.15
ETA_WATER = 0.692e-3
CROWDING_FACTOR_DEFAULT = 3.0
N_A = 6.022e23
DELTA_COMP_MW_MIDPOINT_DEFAULT = 40000.0
DELTA_COMP_SLOPE = 0.001
S_NEUTRAL = 1.0


# ── Logging Setup ───────────────────────────────────────────────────────

def setup_task_logger(name: str, log_dir: Path, task_tag: str) -> logging.Logger:
    """Create a logger that writes to an independent log file per task."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}.{task_tag}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(
            log_dir / f"{task_tag}.log", mode="w", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ── CRLF-safe I/O ──────────────────────────────────────────────────────

def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with forced CRLF → LF normalisation."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), **kwargs)


def load_pickle_safe(path: Path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Compartment Normaliser ─────────────────────────────────────────────

def nc(comp):
    if not comp:
        return "Unknown"
    c = str(comp).strip().lower()
    if "mitochond" in c:
        return "Mitochondria"
    if "cytoplasmic membrane" in c:
        return "Membrane"
    if "membrane" in c:
        return "Membrane"
    if "cytosol" in c or "cytoplasm" in c:
        return "Cytoplasm"
    if "nucle" in c:
        return "Nucleus"
    return str(comp).strip()


# ── Three-Layer Physics ────────────────────────────────────────────────

def R_nm(mw):
    return 0.066 * (mw ** (1.0 / 3.0))


def D_coeff(r, T=T_DEFAULT, eta=None):
    if eta is None:
        eta = ETA_WATER * CROWDING_FACTOR_DEFAULT
    return K_B * T / (6.0 * math.pi * eta * r * 1e-9)


def smol_k(mw1, mw2, T=T_DEFAULT, eta=None):
    if eta is None:
        eta = ETA_WATER * CROWDING_FACTOR_DEFAULT
    r1, r2 = R_nm(mw1), R_nm(mw2)
    d1, d2 = D_coeff(r1, T, eta), D_coeff(r2, T, eta)
    return 4.0 * math.pi * (d1 + d2) * ((r1 + r2) * 1e-9) * N_A


def J_ij(mw1, mw2, c1_uM, c2_uM, T=T_DEFAULT, eta=None):
    return smol_k(mw1, mw2, T, eta) * c1_uM * 1e-6 * c2_uM * 1e-6


def delta_comp(comp_i, comp_j, mw_i, mw_j, midpoint=DELTA_COMP_MW_MIDPOINT_DEFAULT):
    c_i, c_j = nc(comp_i), nc(comp_j)
    if c_i == c_j:
        return 1.0
    min_mw = min(mw_i, mw_j)
    exp_val = DELTA_COMP_SLOPE * (min_mw - midpoint)
    exp_val = max(min(exp_val, 500), -500)
    return 1.0 / (1.0 + math.exp(exp_val))


# ── Data Loading ────────────────────────────────────────────────────────

SPECIES_MAP = {
    "Escherichia coli K-12": "E.coli",
    "Homo sapiens": "Mammalian",
    "Saccharomyces cerevisiae": "Yeast",
}


def load_master(data_dir):
    """Load V41_Master.csv with CRLF safety."""
    path = Path(data_dir) / "V41_Master.csv"
    df = read_csv_safe(path)
    df["sp"] = df["Species"].map(SPECIES_MAP)
    return df


def load_s_values(data_dir):
    path = Path(data_dir) / "s_values.pkl"
    return load_pickle_safe(path)


def load_known_pairs(data_dir):
    path = Path(data_dir) / "known_pairs.pkl"
    return load_pickle_safe(path)


def to_csv_safe(df, path, **kwargs):
    """to_csv with lineterminator/line_terminator compatibility (pandas 1.x vs 2.x)."""
    kw = dict(index=False, **kwargs)
    try:
        return df.to_csv(path, **kw, line_terminator="\n")
    except TypeError:
        kw.pop("line_terminator", None)
        return df.to_csv(path, **kw, lineterminator="\n")


# ── Scoring Engine ──────────────────────────────────────────────────────

def score_all_pairs(anchor_row, partner_df, s_dict, species,
                    T=T_DEFAULT, eta=None,
                    psi_midpoint=DELTA_COMP_MW_MIDPOINT_DEFAULT,
                    noise_psi=None, noise_j=None, noise_s=None):
    """
    Score anchor vs all partners.
    Returns (j_scores, tl_scores, partner_names).
    """
    a_mw = float(anchor_row["MW_Da"])
    a_conc = float(anchor_row["Conc_uM"])
    a_comp = str(anchor_row["Compartment"])
    a_name = str(anchor_row["Node_Name"])

    n = len(partner_df)
    j_scores = np.zeros(n)
    tl_scores = np.zeros(n)
    names = []

    for i, (_, p) in enumerate(partner_df.iterrows()):
        p_mw = float(p["MW_Da"])
        p_conc = float(p["Conc_uM"])
        p_comp = str(p["Compartment"])
        p_name = str(p["Node_Name"])
        names.append(p_name)

        psi = delta_comp(a_comp, p_comp, a_mw, p_mw, psi_midpoint)
        j = J_ij(a_mw, p_mw, a_conc, p_conc, T, eta)

        key = (a_name, p_name, species)
        s = s_dict.get(key, S_NEUTRAL)

        if noise_psi is not None:
            psi *= noise_psi[i]
        if noise_j is not None:
            j *= noise_j[i]
        if noise_s is not None:
            s *= noise_s[i]

        j_scores[i] = psi * j
        tl_scores[i] = psi * j * s

    return j_scores, tl_scores, names


def compute_delta_rank(j_scores, tl_scores):
    """rank_j - rank_tl   (positive = rescue by S-layer)."""
    rank_j = rankdata(-j_scores, method="min")
    rank_tl = rankdata(-tl_scores, method="min")
    return rank_j - rank_tl


def compute_rri(delta_ranks, known_mask):
    """Rank Rescue Index: mean delta_rank for known partners."""
    if known_mask.sum() == 0:
        return np.nan
    return float(np.mean(delta_ranks[known_mask]))


def permutation_test_rri(delta_ranks, known_mask, n_perm=1000, rng=None):
    """Permutation p-value: fraction of random subsets >= observed RRI."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_known = int(known_mask.sum())
    if n_known == 0:
        return np.nan
    obs = np.mean(delta_ranks[known_mask])
    count = 0
    for _ in range(n_perm):
        idx = rng.choice(len(delta_ranks), size=n_known, replace=False)
        if np.mean(delta_ranks[idx]) >= obs:
            count += 1
    return float(count / n_perm)


def hypergeometric_test(n_total, n_known, k_topN, hits_in_topN):
    """
    P-value: probability of observing >= hits_in_topN known partners in top-k by chance.
    Uses scipy.stats.hypergeom.sf (survival function).
    n_total: total candidate pool size
    n_known: number of true known partners in pool
    k_topN: cutoff (e.g. 10 for top 10)
    hits_in_topN: observed hits in top-k
    """
    if n_known == 0 or k_topN <= 0:
        return np.nan
    try:
        from scipy.stats import hypergeom
        # hypergeom.sf(k-1, N, K, n) = P(X >= k)
        # N=pool, K=successes in pool, n=draws, k=observed
        p_val = hypergeom.sf(hits_in_topN - 1, n_total, n_known, min(k_topN, n_total))
        return float(max(p_val, 1e-300))  # avoid exact 0
    except Exception:
        return np.nan


def compute_hits_at_k(tl_scores, known_mask, k=10):
    """
    Fraction of known partners that appear in top-k by TL score.
    Returns (hits_count, hits_ratio).
    """
    if known_mask.sum() == 0:
        return 0, np.nan
    top_idx = np.argsort(-np.asarray(tl_scores))[:k]
    hits = sum(1 for i in top_idx if known_mask[i])
    ratio = hits / known_mask.sum()
    return int(hits), float(ratio)


def global_wilcoxon_test(rri_values):
    """Wilcoxon signed-rank test: are RRI values significantly > 0?

    Returns (statistic, p_value, n_valid).
    """
    from scipy.stats import wilcoxon
    arr = np.asarray(rri_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, len(arr)
    try:
        stat, p = wilcoxon(arr, alternative="greater")
        return float(stat), float(p), len(arr)
    except Exception:
        return np.nan, np.nan, len(arr)
