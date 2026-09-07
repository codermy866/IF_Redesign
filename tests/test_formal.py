from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/formal_v1"


def test_formal_loco_assignments_are_patient_disjoint() -> None:
    path = FORMAL / "fold_assignments.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path, dtype={"case_id": str, "patient_id": str})
    assert len(frame) == 984 * 5
    assert frame.fold.nunique() == 5
    assert (frame.groupby("case_id").size() == 5).all()
    for _, fold in frame.groupby("fold"):
        assert set(fold.split) == {"train", "val", "test"}
        patient_splits = fold.groupby("patient_id").split.nunique()
        assert (patient_splits == 1).all()
        held_centres = fold.loc[fold.split == "test", "center"].unique()
        assert len(held_centres) == 1
        assert not fold.loc[fold.split != "test", "center"].isin(held_centres).any()


def test_formal_prompts_have_no_assistant_or_missing_images() -> None:
    path = FORMAL / "all/evidence_prompts.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 984
    for row in rows:
        assert all(message["role"] != "assistant" for message in row["messages"])
        assert all(Path(image).is_file() for image in row["images"])
        rendered = json.dumps(row["messages"], ensure_ascii=False)
        assert "binary_label" not in rendered
        assert "pseudo_report" not in rendered


def test_formal_variant_matrix_is_complete() -> None:
    if not (FORMAL / "folds").is_dir():
        return
    variants = {
        "diagnosis_only",
        "without_integration",
        "full_hierarchy",
        "colposcopy_only",
        "oct_only",
        "without_clinical",
        "shuffled_cognition",
    }
    for fold in ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang"):
        variant_root = FORMAL / "folds" / fold / "variants"
        assert variants == {path.name for path in variant_root.iterdir() if path.is_dir()}
        for variant in variants:
            for split in ("train", "val"):
                assert (variant_root / variant / f"{split}_sft.jsonl").is_file()
            for split in ("train", "val", "test", "eval"):
                assert (variant_root / variant / f"{split}_prompt.jsonl").is_file()


def test_formal_visual_targets_are_diagnosis_sanitized() -> None:
    path = FORMAL / "folds/shiyan/variants/full_hierarchy/train_sft.jsonl"
    if not path.is_file():
        return
    forbidden = re.compile(r"(?i)(cin\s*[123]?\+?|hsil|lsil|癌|恶性|高级别|低级别|阳性|阴性|positive|negative)")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assistant = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
        for tag in ("colposcopy_morphology", "oct_microstructure", "integration"):
            match = re.search(rf"<{tag}>(.*?)</{tag}>", assistant)
            assert match
            cleaned = match.group(1).replace("[诊断性表述已移除]", "")
            assert not forbidden.search(cleaned)
