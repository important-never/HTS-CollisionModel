# HTS-CollisionModel

Companion code and processed data for:

**Protein specificity reshapes molecular encounters in cellular metabolism**

Tong and Chen, submitted.

This repository supports the analyses in the manuscript. The study asks how expressed proteins and metabolites become prioritized for productive molecular encounters in crowded cells. The model combines three layers into a pairwise protein-metabolite encounter score:

```text
A_ij = Psi_ij x J_ij x S_ij
where Psi represents compartmental access, J represents Smoluchowski collision frequency, and S represents enzyme specificity. The model uses predefined physical and biochemical rules, with no parameters fitted to known partners or downstream outcomes.

What this repository contains
The repository contains code and processed inputs used to reproduce the main computational analyses, including:

construction of the three-layer encounter score;
collision-only and full-model ranking analyses;
Rank Rescue Index (RRI) calculations;
specificity-layer shuffle controls;
leave-one-out and noise-sensitivity analyses;
cross-system rank-shift comparisons;
figure-generation and audit scripts.
The analyses use public biochemical, proteomic, metabolomic, and genome-scale metabolic-model resources, including PaxDb, BRENDA, SABIO-RK, KEGG, UniProt, BiGG Models, and COBRApy-compatible genome-scale metabolic models, as described in the manuscript.

Quick start
git clone https://github.com/important-never/HTS-CollisionModel.git
cd HTS-CollisionModel

# Install Python dependencies
pip install -r requirements.txt

# Run a quick reproducibility check
python reproduce_all.py --quick

# Run the full reproduction workflow
python reproduce_all.py
If the full workflow is not needed, individual analysis scripts can be run from the scripts/ and figures/ folders.

Repository structure
HTS-CollisionModel/
├── reproduce_all.py          # Main reproducibility entry point
├── requirements.txt          # Python dependencies
├── data/                     # Processed input tables and model files
├── scripts/                  # Core analysis scripts
├── figures/                  # Figure and audit scripts
├── results/                  # Generated analysis outputs
├── figures_output/           # Generated figure outputs
└── paper/                    # Manuscript-related source files, if included
Some generated folders may be absent until the reproduction workflow is run.

Main analyses
The code reproduces the following analysis groups:

Encounter model construction
Combines compartmental access, collision frequency, and specificity into pairwise protein-metabolite scores.

Collision baseline analysis
Tests how strongly collision frequency alone aligns with metabolic priority.

Specificity-driven rank-shift analysis
Measures how enzyme specificity changes pair rankings relative to the collision-only baseline.

Focal cofactor analyses
Evaluates rank-shift patterns for GTP, acetyl-CoA, and ATP across the tested systems.

Control and robustness analyses
Includes specificity-layer shuffling, leave-one-out removal, parameter scans, and log-normal noise perturbations.

Cross-system comparison
Compares directional rank shifts across bacterial, yeast, and mammalian systems using shared pair anchors.

Data and reproducibility notes
This repository contains processed data tables and scripts intended to support peer review and reproducibility. The manuscript should be consulted for full details on:

source databases and versions;
preprocessing rules;
fallback rules for specificity values;
mammalian composite-system construction;
statistical tests and random seeds;
limitations of the model and data coverage.
No private login credentials or personal information are required to access or run the repository.

Important interpretation boundary
The model is designed to quantify molecular encounter priority. It does not claim to simulate the entire cell, prove causal mechanisms by experiment, or predict gene essentiality. Model outputs should be interpreted as specificity-aware encounter rankings derived from public biochemical and cellular measurements.

Citation
If you use this code or processed data, please cite the accompanying manuscript once available:

Tong, Z. and Chen, F. Protein specificity reshapes molecular encounters in cellular metabolism.

License
This repository is released under the Apache-2.0 license.

Contact
For questions about the repository, please open an issue on GitHub or contact the corresponding author listed in the manuscript.
