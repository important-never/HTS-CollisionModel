# HTS: Hierarchical Thermo-Spatial Molecular Encounter Model

> **Paper**: *Toward a networked Central Dogma: physical constraints predict molecular encounter specificity across three kingdoms*

A zero-free-parameter physical model (Ψ × J × S) that predicts molecular encounter propensity in living cells. Validated across *E. coli*, *S. cerevisiae*, and mammalian cells, the model identifies three hub molecules (GTP, Acetyl-CoA, ATP) whose enzymatic rescue is conserved over ~2 billion years of evolution.

---

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/HTS-model.git
cd HTS-model

# Install dependencies (Python >= 3.9)
pip install -r requirements.txt

# Quick smoke test (~2 min)
python reproduce_all.py --quick

# Full reproduction (~30-60 min)
python reproduce_all.py

# Cross-species audit only (~5 min)
python reproduce_all.py --audit
```

## What It Reproduces

| Experiment | Paper Section | Output |
|-----------|---------------|--------|
| Metabolite Atlas (HPC-1) | §2.1 S-layer validation | `results/hpc1_metabolite_atlas.csv` |
| S-Shuffle Causality (HPC-7) | §2.1, Table 1 | `results/hpc7_s_shuffle.csv` |
| Cross-Species Audit | §2.4 Global conservation | `results/cross_species_summary.csv` |
| Breaking Point (HPC-8) | §2.4, Table 1 | `results/hpc8_breaking_point.csv` |

A full reproduction report is saved to `REPRODUCTION_REPORT.txt`.

## Repository Structure

```
HTS-model/
├── reproduce_all.py          # One-click entry point
├── requirements.txt          # Python dependencies
├── data/
│   ├── V41_Master.csv        # 437 nodes (proteins + metabolites) × 3 species
│   ├── s_values.pkl          # Biological selectivity layer (S) values
│   ├── s_source.pkl          # S-value provenance (BRENDA / KEGG / literature)
│   ├── known_pairs.pkl       # Ground-truth enzyme-substrate pairs per species
│   ├── GEM_models/           # Genome-scale metabolic models (3 core species)
│   │   ├── iML1515.json      # E. coli K-12
│   │   ├── iMM904.json       # S. cerevisiae
│   │   └── Recon3D.json      # H. sapiens (mammalian proxy)
│   ├── tables/               # Evidence tables (CSV)
│   └── task_lists/           # HPC experiment task definitions
├── scripts/
│   ├── hpc_common.py         # Shared physics engine (Ψ, J, S formulas)
│   ├── hpc1_metabolite_atlas.py
│   ├── hpc1_protein_anchor.py
│   ├── hpc2_monte_carlo.py
│   ├── hpc3_phase_space.py
│   ├── hpc4_gem_cascade.py
│   ├── hpc6_loo_sensitivity.py
│   ├── hpc7_s_shuffle.py
│   ├── hpc8_lognormal_mc.py
│   ├── prepare_data.py       # Data preparation pipeline
│   └── merge_results.py      # Merge HPC outputs
├── figures/
│   ├── gen_fig2_combined.py   # Figure 2 generation
│   ├── gen_fig3_combined.py   # Figure 3 generation
│   ├── cross_species_rri_audit.py  # Auditable cross-species correlations
│   └── *.csv                  # Audit output data
├── paper/
│   ├── main.tex               # Manuscript source
│   └── supplementary_information.tex
├── results/                   # (generated) Experiment outputs
└── figures_output/            # (generated) Regenerated figures
```

## Three-Layer Model

The model computes encounter propensity between molecules *i* and *j* as:

**A_ij = Ψ_ij × J_ij × S_ij**

| Layer | Name | Physical meaning | Parameters |
|-------|------|-----------------|------------|
| Ψ | Compartment gating | Same compartment = 1; cross-compartment sigmoid decay | MW, compartment annotations |
| J | Smoluchowski collision frequency | Diffusion-limited encounter rate | MW → radius → diffusion coeff × concentration |
| S | Biological selectivity | Evolutionary enzyme specificity | kcat/Km (BRENDA), KEGG co-occurrence |

All physical constants are derived from first principles (T = 310.15 K, η = 3× water viscosity at 37°C). The model has **zero free parameters**.

## Key Findings

- **GTP**: Nucleotide-specific rescue (bootstrap *p* ≤ 0.01 vs ATP/CTP/UTP), signalling currency
- **Acetyl-CoA**: Most physically hidden yet most rescued metabolite (RRI = 41.6 ± 4.7), co-elevates entire redox module
- **ATP**: Hub ranking curse with precise ATP synthase extraction (Δrank = +24 to +41)
- **Cross-species conservation**: RRI Spearman ρ = 0.67–0.84 across ~2 billion years (n = 3,422–3,660)
- **S-layer causality**: Shuffling S-values collapses RRI from 20–30 to 2–6 (Wilcoxon *p* < 10⁻³⁷)

## Data Sources

| Data | Source | Reference |
|------|--------|-----------|
| Protein abundance | PaxDb 5.0 | Wang et al., 2015 |
| Metabolite concentrations | Park et al., 2016; Bennett et al., 2009 | |
| Enzyme kinetics (kcat/Km) | BRENDA 2026.1 | Chang et al., 2021 |
| Metabolic models | BiGG Models | King et al., 2016 |

## License

This repository accompanies a research paper under review. Code is provided for reproducibility and peer review. Please cite the paper if you use this work.

## Contact

For questions about the model or data, please open an issue on this repository.
