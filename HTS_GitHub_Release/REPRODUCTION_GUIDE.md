# Reproduction Guide

This guide explains how to reproduce the computational results reported in:

> **Toward a networked Central Dogma: protein specificity reshapes molecular encounters across evolution**
> Zhenzhe Tong and Fei Chen, Hainan University

---

## System Requirements

### Hardware
- Any modern computer with ≥8 GB RAM
- Full HPC experiments (25,727 tasks) require a 64-core cluster; single-machine equivalents are provided

### Software
- **Python** ≥ 3.9 (tested on 3.11 and 3.13)
- **pip** (Python package manager)
- **Git** (optional, for cloning)

### Operating System
- Tested on Windows 10/11 and Ubuntu 22.04
- Should work on any OS with Python support

---

## Installation

### Step 1: Clone or download

```bash
git clone https://github.com/<your-username>/HTS-model.git
cd HTS-model
```

Or download and extract the ZIP archive.

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.10 | Statistical tests (Wilcoxon, Spearman) |
| pandas | ≥ 1.5 | Data manipulation |
| matplotlib | ≥ 3.7 | Figure generation |
| scikit-learn | ≥ 1.2 | AUC computation |

No GPU is required. No external databases need to be downloaded; all necessary data are included in `data/`.

---

## Reproduction

### Quick smoke test (~2 min)

Runs all four experiments on the three star molecules only (GTP, Acetyl-CoA, ATP):

```bash
python reproduce_all.py --quick
```

### Full reproduction (~20–40 min)

Runs all experiments on the complete metabolite and protein sets:

```bash
python reproduce_all.py
```

### Cross-species audit only (~5 min)

Reproduces the cross-species RRI correlation analysis (paper Section 2.4):

```bash
python reproduce_all.py --audit
```

### What gets reproduced

| Experiment | Paper reference | Script equivalent |
|------------|----------------|-------------------|
| Three-layer scoring (Ψ × J × S) | Table 1, Fig. 2c | HPC-1 |
| S-shuffle causality test | Table 1 (S-shuffle Δ) | HPC-7 |
| Cross-species RRI correlation | Section 2.4, Fig. 3d | HPC-1 × 3 species |
| Log-normal breaking point | Table 1 (σ_break) | HPC-8 |

### Output

All results are written to the `results/` directory (created automatically):

```
results/
├── exp1_metabolite_atlas.csv
├── exp2_s_shuffle.csv
├── exp3_cross_species_audit.csv
├── exp4_lognormal_breaking.csv
└── REPRODUCTION_REPORT.txt
```

`REPRODUCTION_REPORT.txt` contains a summary of all experiments with pass/fail status.

---

## Repository Structure

```
HTS_GitHub_Release/
├── reproduce_all.py          # One-click reproduction entry point
├── requirements.txt          # Python dependencies
├── REPRODUCTION_GUIDE.md     # This file
├── README.md                 # Project overview
├── LICENSE                   # MIT License
├── data/
│   ├── V41_Master.csv        # Master data (437 nodes × 3 species)
│   ├── s_values.pkl          # S-layer values
│   ├── s_source.pkl          # S-layer provenance
│   ├── known_pairs.pkl       # Ground-truth enzyme–substrate pairs
│   ├── GEM_models/           # Genome-scale metabolic models
│   │   ├── iML1515.json      #   E. coli (Monk et al. 2017)
│   │   ├── iMM904.json       #   S. cerevisiae (Mo et al. 2009)
│   │   └── Recon3D.json      #   H. sapiens (Brunk et al. 2018)
│   └── tables/               # Pre-computed evidence tables
├── scripts/                  # HPC experiment scripts (reference)
│   ├── hpc_common.py         #   Shared utilities
│   ├── prepare_data.py       #   Data preparation
│   ├── hpc1_metabolite_atlas.py
│   ├── hpc1_protein_anchor.py
│   ├── hpc2_monte_carlo.py
│   ├── hpc3_phase_space.py
│   ├── hpc4_gem_cascade.py   #   Gene essentiality (requires cobra)
│   ├── hpc6_loo_sensitivity.py
│   ├── hpc7_s_shuffle.py
│   ├── hpc8_lognormal_mc.py
│   └── merge_results.py
└── results/                  # Output directory (populated after run)
```

---

## Notes

- The `results/` directory is initially empty and is populated after running `reproduce_all.py`.
- The four experiments in `reproduce_all.py` cover the complete computational pipeline of the paper (model scoring, causality, cross-species conservation, robustness). Additional validation experiments (gene essentiality via FBA, Newman modularity) require the `cobra` package and are provided as standalone HPC scripts in `scripts/` for reference.
- Full HPC experiments (25,727 tasks across 8 experiment types on a 64-core cluster) are documented in `scripts/`. Contact the authors for raw HPC output data.
- All random seeds are fixed (`random.seed(42)`, `np.random.seed(42)`) for exact reproducibility.
- No results are hardcoded; all values are computed from the raw data at runtime.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/` | Ensure you are running from the repository root |
| Slow execution | Use `--quick` for a 2-minute smoke test |
| Different numeric values | Minor floating-point differences across platforms are expected; all qualitative conclusions should match |

---

## Contact

For questions about the code or data, please contact:
- Fei Chen: feichen@hainanu.edu.cn
