#!/usr/bin/env python3
"""Build a case-intrinsic OCT-position cluster manifest for ICES-v1.

No label is used to decide which images enter a cluster. The output keeps paths
only in this local internal manifest; the later feature cache must contain no
raw paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cervix_cogalign.io import existing_paths, stable_score, write_jsonl  # noqa: E402
from cervix_cogalign.sequence_evidence import cluster_oct_position_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config["manifest"], dtype={"case_id": str}).set_index("case_id", drop=False)
    assignment = pd.read_csv(config["fold_assignments"], dtype={"case_id": str, "patient_id": str})
    included_ids = set(assignment.case_id.astype(str))
    source = manifest.loc[manifest.index.intersection(included_ids)].copy()
    if set(source.case_id.astype(str)) != included_ids:
        raise AssertionError("ICES cluster cohort must exactly match the frozen LOCO assignment")
    if source.patient_id.duplicated().any():
        raise AssertionError("ICES cluster cohort requires one selected case per patient")

    rows = []
    colposcopy_counts = []
    position_counts = []
    for raw in source.to_dict(orient="records"):
        oct_paths = existing_paths(raw["oct_paths"])
        colposcopy_paths = sorted(existing_paths(raw["colposcopy_paths"]))
        if not oct_paths or not colposcopy_paths:
            raise AssertionError("Frozen ICES cohort contains an unavailable required modality")
        clusters = cluster_oct_position_frames(
            oct_paths,
            expected_positions=int(config["sequence_units"]["expected_oct_positions"]),
            expected_frames_per_position=int(config["sequence_units"]["expected_frames_per_oct_position"]),
        )
        colposcopy_counts.append(len(colposcopy_paths))
        position_counts.append(len(clusters))
        units = [{"unit_id": "clinical", "kind": "clinical", "fields": ["age", "hpv", "tct"]}]
        units.extend(
            {"unit_id": f"colposcopy_view_{index:02d}", "kind": "colposcopy_view", "path": path}
            for index, path in enumerate(colposcopy_paths)
        )
        units.extend(
            {
                "unit_id": cluster.unit_id,
                "kind": "oct_position_cluster",
                "scanner_position": [cluster.c_index, cluster.s_index],
                "frame_paths": list(cluster.frame_paths),
            }
            for cluster in clusters
        )
        rows.append(
            {
                "id": str(raw["case_id"]),
                "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
                "center": str(raw["hospital_name"]),
                "label": int(raw["binary_label"]),
                "evidence_units": units,
            }
        )
    if len(rows) != len(included_ids) or len({row["patient_id"] for row in rows}) != len(rows):
        raise AssertionError("ICES manifest lost case/patient uniqueness")
    destination = output / "cluster_manifest.jsonl"
    write_jsonl(destination, rows)
    audit = {
        "experiment_id": config["experiment_id"],
        "n_cases": len(rows),
        "n_patients": len(rows),
        "oct_position_clusters_per_case": dict(sorted(Counter(position_counts).items())),
        "colposcopy_views_per_case": dict(sorted(Counter(colposcopy_counts).items())),
        "label_used_for_unit_construction": False,
        "claim_boundary": "C/S is a reproducible scanner-position grouping parsed from filenames; it is not a clinician-confirmed lesion or anatomical coordinate.",
    }
    (output / "cluster_manifest_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
