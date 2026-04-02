#!/usr/bin/env python3
"""
Data Preparation — Full Pipeline Extraction  (V3 — HPC-architect grade)
========================================================================
Runs the FULL three_layer_model.py S-layer builder and generates
comprehensive known_pairs for metabolite-PROTEIN interactions.

Usage (Windows):
  D:\\tools\\python.exe scripts\\prepare_data.py ^
      --source_dir "...\\V4_Clean_Data" --output_dir data

Usage (Linux):
  python scripts/prepare_data.py \\
      --source_dir /path/to/V4_Clean_Data --output_dir data
"""
import argparse
import csv
import io
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _to_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw, line_terminator="\n")
    except TypeError:
        kw.pop("line_terminator", None)
        df.to_csv(path, index=False, **kw, lineterminator="\n")


def read_csv_crlf_safe(path: Path):
    """Read CSV rows as list-of-dicts, forcing CRLF → LF."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


SPECIES_MAP = {
    "Escherichia coli K-12": "E.coli",
    "Homo sapiens": "Mammalian",
    "Saccharomyces cerevisiae": "Yeast",
}

KNOWN_SOURCES = {
    "Literature kcat/Km",
    "BRENDA kcat/Km",
    "BRENDA kcat/Km(calc)",
    "KEGG_GT (enzyme-substrate)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--master_csv", default="V41_Master.csv", help="Master CSV filename (e.g. V42_Master.csv)")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    src = args.source_dir.resolve()

    # ── 1. Copy master CSV ───────────────────────────────────────────────
    master_src = src / args.master_csv
    master_dst = out / "V41_Master.csv"
    print(f"[1/7] Copying {args.master_csv} -> V41_Master.csv ...")
    shutil.copy2(master_src, master_dst)
    print(f"       → {master_dst}")

    # ── 2. Copy GEM models ──────────────────────────────────────────────
    gem_src = src / "GEM_models"
    gem_dst = out / "GEM_models"
    if gem_src.exists():
        gem_dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in gem_src.iterdir():
            if f.suffix == ".json":
                shutil.copy2(f, gem_dst / f.name)
                copied += 1
        print(f"[2/7] Copied {copied} GEM model(s)")

    # ── 3. Load master data ─────────────────────────────────────────────
    print(f"[3/7] Loading master data ...")
    all_rows = read_csv_crlf_safe(master_src)
    nodes_by_species = {}
    for row in all_rows:
        sp = SPECIES_MAP.get(row.get("Species"))
        if sp is None:
            continue
        nodes_by_species.setdefault(sp, []).append(row)

    for sp, nodes in nodes_by_species.items():
        n_prot = sum(1 for n in nodes if n["Category"] == "Protein")
        n_met = sum(1 for n in nodes if n["Category"] == "Metabolite")
        print(f"       {sp}: {len(nodes)} nodes ({n_prot} prot, {n_met} met)")

    # ── 4. Import and run the full three-layer S-layer pipeline ─────────
    print(f"\n[4/7] Building FULL S-layer via three_layer_model pipeline ...")
    sys.path.insert(0, str(src))
    saved_cwd = os.getcwd()
    os.chdir(str(src))

    try:
        from smoluchowski_null_model import (
            METABOLITE_KEGG, build_kegg_ground_truth,
        )
        from three_layer_model import (
            build_s_layer, build_name_lookup, parse_brenda_json,
            LITERATURE_KCAT_KM, MANUAL_EC,
            BRENDA_JSON, KEGG_MAP, KEGG_RXN_CACHE,
        )

        kegg_mapping = {}
        kegg_map_path = Path(KEGG_MAP)
        if kegg_map_path.exists():
            kegg_mapping = json.loads(kegg_map_path.read_text(encoding="utf-8"))
            n_ge = len(kegg_mapping.get("gene_to_ec", {}))
            print(f"       KEGG mapping loaded: {n_ge} gene→EC entries")

        all_nodes = []
        for sp in ["E.coli", "Mammalian", "Yeast"]:
            all_nodes.extend(nodes_by_species.get(sp, []))

        met_rows = [n for n in all_nodes if n["Category"] == "Metabolite"]
        kegg_to_names, name_to_kegg = build_name_lookup(kegg_mapping, met_rows)

        gene_to_ec = kegg_mapping.get("gene_to_ec", {})
        for gid, ecs in MANUAL_EC.items():
            if gid not in gene_to_ec or not gene_to_ec[gid]:
                gene_to_ec[gid] = ecs

        target_ecs = set()
        for n in all_nodes:
            if n["Category"] == "Protein":
                target_ecs.update(gene_to_ec.get(n.get("KEGG_ID", ""), []))
        print(f"       Target ECs: {len(target_ecs)}")

        kinetics_records = []
        brenda_path = Path(BRENDA_JSON)
        if brenda_path.exists():
            target_organisms = [
                "Escherichia coli", "Homo sapiens",
                "Saccharomyces cerevisiae", "Mus musculus",
            ]
            kinetics_records = parse_brenda_json(
                str(brenda_path), sorted(target_ecs),
                kegg_to_names, target_organisms,
            )
            print(f"       BRENDA kinetics records: {len(kinetics_records)}")
        else:
            print(f"       WARNING: BRENDA JSON not found at {brenda_path}")

        # Build S-layer PER SPECIES
        s_dict_full = {}
        s_source_full = {}

        for sp in ["E.coli", "Mammalian", "Yeast"]:
            nodes = nodes_by_species.get(sp, [])
            if not nodes:
                continue
            print(f"\n       Building S-layer for {sp} ...")
            kegg_gt = build_kegg_ground_truth(nodes)
            s_vals, s_src = build_s_layer(
                nodes, kegg_mapping, kinetics_records, kegg_gt,
            )
            n_nonneutral = sum(1 for v in s_vals.values() if v != 1.0)
            print(f"         {len(s_vals)} pairs, {n_nonneutral} non-neutral")

            for key, val in s_vals.items():
                s_dict_full[(key[0], key[1], sp)] = val
                s_dict_full[(key[1], key[0], sp)] = val
                src_label = s_src.get(key, "unknown")
                s_source_full[(key[0], key[1], sp)] = src_label
                s_source_full[(key[1], key[0], sp)] = src_label

    except Exception:
        os.chdir(saved_cwd)
        print("\n  FATAL ERROR during S-layer build:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    os.chdir(saved_cwd)
    print(f"\n       Total S-layer entries (bidirectional): {len(s_dict_full)}")

    s_path = out / "s_values.pkl"
    with open(s_path, "wb") as f:
        pickle.dump(s_dict_full, f, protocol=4)
    print(f"       Saved → {s_path}")

    ss_path = out / "s_source.pkl"
    with open(ss_path, "wb") as f:
        pickle.dump(s_source_full, f, protocol=4)

    # ── 5. Build known_pairs from S-layer evidence ──────────────────────
    #   Key fix: extract metabolite-PROTEIN pairs with direct BRENDA/
    #   Literature/KEGG_GT evidence, not just metabolite-metabolite
    #   co-reaction pairs from build_kegg_ground_truth.
    print(f"\n[5/7] Building known_pairs from S-layer evidence ...")
    known_pairs = {}
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        nodes = nodes_by_species.get(sp, [])
        node_names = {n["Node_Name"] for n in nodes}
        prot_names = {n["Node_Name"] for n in nodes if n["Category"] == "Protein"}
        met_names = {n["Node_Name"] for n in nodes if n["Category"] == "Metabolite"}

        pairs = set()
        for (n1, n2, s), src_label in s_source_full.items():
            if s != sp:
                continue
            if n1 not in node_names or n2 not in node_names:
                continue
            is_met_prot = (
                (n1 in met_names and n2 in prot_names) or
                (n1 in prot_names and n2 in met_names)
            )
            if not is_met_prot:
                continue
            prefix = src_label.split(" ")[0] if src_label else ""
            full_match = any(src_label.startswith(k) for k in KNOWN_SOURCES)
            if full_match:
                pairs.add((n1, n2))
                pairs.add((n2, n1))

        known_pairs[sp] = pairs
        n_unique = len(pairs) // 2
        print(f"       {sp}: {n_unique} known met-prot pairs")

        for anchor in ["GTP", "Acetyl-CoA", "ATP"]:
            n_kp = sum(1 for p in pairs if p[0] == anchor) // 1
            anchor_kp = {p[1] for p in pairs if p[0] == anchor}
            print(f"         {anchor}: {len(anchor_kp)} known protein partners")

    kp_path = out / "known_pairs.pkl"
    with open(kp_path, "wb") as f:
        pickle.dump(known_pairs, f, protocol=4)
    print(f"       Saved → {kp_path}")

    # ── 6. Create task lists ────────────────────────────────────────────
    print(f"\n[6/7] Creating task lists ...")
    import pandas as pd

    hpc1_tasks = []
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        for n in nodes_by_species.get(sp, []):
            if n["Category"] == "Metabolite":
                hpc1_tasks.append({"species": sp, "anchor": n["Node_Name"]})
    hpc1_path = out / "hpc1_task_list.csv"
    _to_csv(pd.DataFrame(hpc1_tasks), hpc1_path)
    print(f"       hpc1_task_list.csv: {len(hpc1_tasks)} tasks")

    hpc1p_tasks = []
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        for n in nodes_by_species.get(sp, []):
            if n["Category"] == "Protein":
                hpc1p_tasks.append({"species": sp, "anchor": n["Node_Name"]})
    hpc1p_path = out / "hpc1p_task_list.csv"
    _to_csv(pd.DataFrame(hpc1p_tasks), hpc1p_path)
    print(f"       hpc1p_task_list.csv: {len(hpc1p_tasks)} tasks (protein-anchor)")

    # HPC-6 task list: (species, remove_idx) for non-neutral S entries
    hpc6_tasks = []
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        sp_entries = sorted(
            k for k, v in s_dict_full.items()
            if len(k) == 3 and k[2] == sp and v != 1.0 and v > 0
        )
        seen = set()
        idx = 0
        for key in sp_entries:
            canonical = tuple(sorted([key[0], key[1]])) + (sp,)
            if canonical in seen:
                continue
            seen.add(canonical)
            hpc6_tasks.append({"species": sp, "remove_idx": idx})
            idx += 1
    hpc6_path = out / "hpc6_task_list.csv"
    _to_csv(pd.DataFrame(hpc6_tasks), hpc6_path)
    print(f"       hpc6_task_list.csv: {len(hpc6_tasks)} tasks")

    # HPC-4 task list: (species, met_idx) for GEM cytoplasmic metabolites
    hpc4_tasks = []
    gem_dst = out / "GEM_models"
    gem_species_files = {
        "E.coli": "iML1515.json",
        "Yeast": "iMM904.json",
        "Mammalian": "Recon3D.json",
    }
    try:
        import cobra as _cobra
        for sp, gem_file in gem_species_files.items():
            gem_path = gem_dst / gem_file
            if not gem_path.exists():
                print(f"       WARNING: {gem_path} not found, skipping {sp} HPC-4 tasks")
                continue
            model = _cobra.io.load_json_model(str(gem_path))
            cyto_mets = sorted(
                [m for m in model.metabolites
                 if m.id.lower().endswith("_c") or "[c]" in m.id.lower()],
                key=lambda x: x.id,
            )
            for idx in range(len(cyto_mets)):
                hpc4_tasks.append({"species": sp, "met_idx": idx})
            print(f"       {sp}: {len(cyto_mets)} cytoplasmic metabolites for HPC-4")
    except ImportError:
        print("       WARNING: cobra not installed, generating HPC-4 task list with estimated counts")
        est_counts = {"E.coli": 800, "Yeast": 600, "Mammalian": 2500}
        for sp, n in est_counts.items():
            for idx in range(n):
                hpc4_tasks.append({"species": sp, "met_idx": idx})
        print(f"       (estimated total: {len(hpc4_tasks)} tasks -- exact count determined at runtime)")

    hpc4_path = out / "hpc4_task_list.csv"
    _to_csv(pd.DataFrame(hpc4_tasks), hpc4_path)
    print(f"       hpc4_task_list.csv: {len(hpc4_tasks)} tasks")

    # HPC-7 task list: S-shuffle causality (species, anchor, seed)
    # 3 species × 3 star anchors × 100 seeds = 900 tasks
    STAR_ANCHORS = ["GTP", "Acetyl-CoA", "ATP"]
    hpc7_tasks = []
    for sp in SPECIES:
        for anc in STAR_ANCHORS:
            for seed in range(100):
                hpc7_tasks.append({"species": sp, "anchor": anc, "seed": seed})
    hpc7_path = out / "hpc7_task_list.csv"
    _to_csv(pd.DataFrame(hpc7_tasks), hpc7_path)
    print(f"       hpc7_task_list.csv: {len(hpc7_tasks)} tasks")

    # HPC-8 task list: log-normal noise MC (species, anchor, sigma, seed)
    # 3 species × 3 anchors × 6 sigmas × 5 seeds = 270 tasks
    SIGMAS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    hpc8_tasks = []
    for sp in SPECIES:
        for anc in STAR_ANCHORS:
            for sigma in SIGMAS:
                for seed in range(5):
                    hpc8_tasks.append({"species": sp, "anchor": anc, "sigma": sigma, "seed": seed})
    hpc8_path = out / "hpc8_task_list.csv"
    _to_csv(pd.DataFrame(hpc8_tasks), hpc8_path)
    print(f"       hpc8_task_list.csv: {len(hpc8_tasks)} tasks")

    # ── 7. Comprehensive verification ───────────────────────────────────
    print(f"\n[7/7] ══════════ VERIFICATION ══════════")
    all_ok = True
    for sp in ["E.coli", "Mammalian", "Yeast"]:
        sp_s = {k: v for k, v in s_dict_full.items() if k[2] == sp}
        n_total = len(sp_s)
        n_nonneutral = sum(1 for v in sp_s.values() if v != 1.0)
        kp = known_pairs.get(sp, set())
        n_known = len(kp) // 2

        status = "OK" if n_known > 5 else "WARN"
        if n_known == 0:
            status = "FAIL"
            all_ok = False
        print(f"  [{status}] {sp}: {n_total} S entries, "
              f"{n_nonneutral} non-neutral, {n_known} known met-prot pairs")

        for anchor in ["GTP", "Acetyl-CoA", "ATP"]:
            anchor_nonneutral = sum(
                1 for k, v in sp_s.items()
                if k[0] == anchor and v != 1.0
            )
            anchor_kp = len({p[1] for p in kp if p[0] == anchor})
            flag = "[Y]" if anchor_kp > 0 else "[N]"
            print(f"    {flag} {anchor}: {anchor_nonneutral} non-neutral S, "
                  f"{anchor_kp} known partners")

    if all_ok:
        print(f"\n  [PASS] ALL CHECKS PASSED -- data package ready for upload")
    else:
        print(f"\n  [FAIL] SOME CHECKS FAILED -- review warnings above")

    print(f"\n{'='*60}")
    print(f"  Data preparation complete!")
    print(f"  Output directory: {out.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
