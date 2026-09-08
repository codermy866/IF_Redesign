#!/usr/bin/env python3
"""Create a claim-bounded inventory of all completed ICES source-only work."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    output = ROOT / "results/ices_v1"
    original = load(output / "source_ablation_summary.json")
    m2 = load(output / "supplement_m2_cost_frontier/m2_cost_frontier_summary.json")
    m3 = load(output / "supplement_clean_m3/clean_m3_supplement_summary.json")
    m1 = load(output / "supplement_m1_frame_sensitivity/m1_frame_sensitivity_summary.json")
    source_gate = original["source_gates"]
    report = {
        "experiment_family": "ICES source-only development and post-screen exploratory supplement",
        "outer_test_labels_opened": False,
        "outer_test_predictions_written": False,
        "original_locked_screen": {
            "n_runs": original["n_runs"], "all_locked_source_gates_pass": source_gate["all_locked_source_gates_pass"],
            "source_gates": source_gate,
        },
        "m2_locked_cost_frontier": m2,
        "m3_clean_component_supplement": m3,
        "m1_reference_frame_sensitivity": m1,
        "allowed_conclusion": "All results are source-train/source-validation development evidence. The original locked screen failed; the finite post-screen supplement is exploratory and cannot authorize a LOCO or clinical claim.",
        "prohibited_conclusion": "Do not describe any result as external validation, causal pathology evidence, lesion localization, prospective acquisition savings, or clinical deployment performance.",
    }
    (output / "ALL_SOURCE_EXPERIMENTS_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ICES all source-only experiments", "",
        "## Bottom line", "",
        "The locked original source screen failed its advancement gate; no outer-centre evaluation was performed. The remaining finite supplement is exploratory and addresses design isolation, not rescue by tuning.", "",
        "## Completed packages", "",
        "- Original M1-M3 screen: 45 runs, complete; gates failed.",
        f"- Locked-margin M2 frontier: complete; adaptive rule non-dominated in {m2['adaptive_non_dominated_runs']['count']}/{m2['adaptive_non_dominated_runs']['total']} source-validation runs.",
        "- Clean M3 component supplement: complete; see its paired source-validation contrasts.",
        "- M1 reference-frame sensitivity: complete; see its frame-1/frame-5/frame-10 comparison.",
        "",
        "No result in this report uses held-out LOCO predictions or labels.",
    ]
    (output / "ALL_SOURCE_EXPERIMENTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(output / "ALL_SOURCE_EXPERIMENTS_SUMMARY.json"), "outer_test_labels_opened": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
