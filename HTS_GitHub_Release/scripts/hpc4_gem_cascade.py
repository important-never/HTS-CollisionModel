#!/usr/bin/env python3
"""
HPC-4: Genome-wide GEM Perturbation Cascade
============================================
ONE task = ONE (species, met_idx) pair.  FBA perturbation + shadow price.
Error-safe with independent logging.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cobra
except ImportError:
    print("ERROR: cobra not installed. Run: pip install cobra", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_common import setup_task_logger, to_csv_safe

GEM_FILES = {
    "E.coli": "iML1515.json",
    "Yeast": "iMM904.json",
    "Mammalian": "Recon3D.json",
}


def get_cytoplasmic_metabolites(model):
    mets = []
    for m in model.metabolites:
        mid = m.id.lower()
        if mid.endswith("_c") or "[c]" in mid:
            mets.append(m)
    return sorted(mets, key=lambda x: x.id)


def perturb_metabolite(model, met, factor, baseline_sol, log):
    baseline_obj = baseline_sol.objective_value
    baseline_fluxes = baseline_sol.fluxes.to_dict()
    shadow = baseline_sol.shadow_prices.get(met.id, 0.0)

    with model as m:
        focus = m.metabolites.get_by_id(met.id)
        affected = []
        for rxn in list(focus.reactions):
            r = m.reactions.get_by_id(rxn.id)
            bf = float(baseline_fluxes.get(r.id, 0.0))
            if abs(bf) < 1e-9:
                continue
            if bf > 0:
                new_ub = bf * factor
                r.upper_bound = min(r.upper_bound, max(new_ub, r.lower_bound))
            else:
                new_lb = bf * factor
                r.lower_bound = max(r.lower_bound, min(new_lb, r.upper_bound))
            affected.append(r.id)

        if not affected:
            return {
                "met_id": met.id, "met_name": met.name,
                "n_reactions": len(list(met.reactions)),
                "shadow_price": float(shadow), "status": "no_active_reactions",
                "n_affected": 0, "baseline_growth": float(baseline_obj),
                "perturbed_growth": float(baseline_obj),
                "growth_impact": 0.0, "mean_flux_shift": 0.0,
            }

        try:
            perturbed = m.optimize()
            perturbed_obj = (
                perturbed.objective_value
                if perturbed.status == "optimal" else 0.0
            )
            perturbed_fluxes = (
                perturbed.fluxes.to_dict()
                if perturbed.status == "optimal" else {}
            )
        except Exception as exc:
            log.warning(f"FBA failed for {met.id}: {exc}")
            perturbed_obj = 0.0
            perturbed_fluxes = {}

        shifts = []
        for rid in affected:
            pf = perturbed_fluxes.get(rid, 0.0)
            bf_val = baseline_fluxes[rid]
            denom = max(abs(bf_val), 1e-6)
            shifts.append(abs(pf - bf_val) / denom)

        growth_impact = (baseline_obj - perturbed_obj) / max(abs(baseline_obj), 1e-9)

    return {
        "met_id": met.id, "met_name": met.name,
        "n_reactions": len(list(met.reactions)),
        "shadow_price": float(shadow), "status": "flux_constrained",
        "n_affected": len(affected), "baseline_growth": float(baseline_obj),
        "perturbed_growth": float(perturbed_obj),
        "growth_impact": float(growth_impact),
        "mean_flux_shift": float(np.mean(shifts)) if shifts else 0.0,
    }


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_met{args.met_idx:04d}"
    log_dir = out_dir.parent / "logs" / "hpc4"
    log = setup_task_logger("hpc4", log_dir, tag)

    data_dir = Path(args.data_dir)
    gem_path = data_dir / "GEM_models" / GEM_FILES[args.species]
    if not gem_path.exists():
        log.error(f"GEM not found: {gem_path}")
        return

    log.info(f"Loading {args.species} GEM ...")
    model = cobra.io.load_json_model(str(gem_path))
    cyto_mets = get_cytoplasmic_metabolites(model)
    log.info(f"  {len(cyto_mets)} cytoplasmic metabolites")

    if args.met_idx >= len(cyto_mets):
        log.warning(f"SKIP  met_idx {args.met_idx} >= {len(cyto_mets)}")
        return

    met = cyto_mets[args.met_idx]
    log.info(f"  Target: {met.id} ({met.name}), factor={args.factor}")

    baseline = model.optimize()
    if baseline.status != "optimal":
        log.error(f"Baseline FBA not optimal ({baseline.status})")
        return

    result = perturb_metabolite(model, met, args.factor, baseline, log)
    result["species"] = args.species
    result["factor"] = args.factor

    csv_path = out_dir / f"hpc4_{tag}.csv"
    to_csv_safe(pd.DataFrame([result]), csv_path)
    log.info(f"DONE   {met.id}: impact={result['growth_impact']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--species", required=True, choices=list(GEM_FILES.keys()))
    ap.add_argument("--met_idx", type=int, required=True)
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        tag = f"{args.species}_met{args.met_idx:04d}"
        err_dir = Path(args.output_dir).parent / "logs" / "hpc4"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{tag}.error").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR  {tag}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
