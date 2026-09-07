from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from cervix_cogalign.counterfactual import (
    gaussian_blur_roi,
    matched_random_bbox,
    normalized_to_pixels,
    proxy_gradient_energy_bbox,
)
from cervix_cogalign.io import evenly_spaced, parse_path_list, stable_score
from cervix_cogalign.parsing import extract_json_object, parse_prediction
from cervix_cogalign.prompts import normalize_hpv, normalize_tct, sanitize_observation
from cervix_cogalign.rewards import cognition_reward, diagnosis_reward, format_reward
from cervix_cogalign.statistics import holm_adjust, paired_bootstrap_delta


def test_deterministic_helpers() -> None:
    assert parse_path_list('["a", "b"]') == ["a", "b"]
    assert evenly_spaced(["a", "b", "c", "d", "e"], 3) == ["a", "c", "e"]
    assert stable_score("x", seed=7) == stable_score("x", seed=7)


def test_clinical_unknown_codes_are_not_overinterpreted() -> None:
    assert normalize_hpv("1") == "状态未知"
    assert normalize_tct("1") == "结果未知"
    assert "16/18" in normalize_hpv("16,52")
    assert normalize_tct("ASC-H") == "ASC-H"


def test_parsing_and_sanitization() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_prediction("<diagnosis>positive</diagnosis><probability>0.8</probability>") == (1, 0.8)
    assert "CIN" not in sanitize_observation("考虑CIN2+，positive")


def test_rewards() -> None:
    text = """<clinical_context>HPV信息但存在局限</clinical_context>
<colposcopy_morphology>部位可见醋白，边界清楚，血管可见</colposcopy_morphology>
<oct_microstructure>上皮分层消失，基底膜不清，信号衰减</oct_microstructure>
<integration>综合证据一致，但仍有不确定</integration>
<diagnosis>positive</diagnosis>
<probability>0.8</probability>"""
    assert format_reward(text) == 1.0
    assert cognition_reward(text) == 1.0
    assert diagnosis_reward(text, "positive") == 1.0
    assert diagnosis_reward(text, "negative") == 0.0


def test_paired_bootstrap_and_holm() -> None:
    import pandas as pd

    common = {
        "id": [str(i) for i in range(8)],
        "patient_id": [f"p{i}" for i in range(8)],
        "center": ["a"] * 4 + ["b"] * 4,
        "label": [0, 0, 1, 1] * 2,
    }
    reference = pd.DataFrame(
        common | {"probability": [0.4, 0.6, 0.4, 0.6] * 2, "predicted_label": [0, 1, 0, 1] * 2}
    )
    candidate = pd.DataFrame(
        common | {"probability": [0.1, 0.2, 0.8, 0.9] * 2, "predicted_label": [0, 0, 1, 1] * 2}
    )
    report = paired_bootstrap_delta(reference, candidate, 50, 7, metrics=("accuracy", "auprc"))
    assert report["n_paired"] == 8
    assert report["candidate_minus_reference"]["pooled"]["accuracy"]["point"] == 0.5
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_counterfactual_only_changes_roi() -> None:
    arr = np.zeros((40, 50, 3), dtype=np.uint8)
    arr[10:30, 15:35] = np.indices((20, 20))[0][..., None] * 10
    image = Image.fromarray(arr)
    box = normalized_to_pixels([0.3, 0.25, 0.7, 0.75], image.size)
    blurred = np.asarray(gaussian_blur_roi(image, box, radius=3))
    original = np.asarray(image)
    outside = np.ones(original.shape[:2], dtype=bool)
    outside[box[1] : box[3], box[0] : box[2]] = False
    assert np.array_equal(blurred[outside], original[outside])
    random_box = matched_random_bbox(box, image.size, seed=3)
    assert (random_box[2] - random_box[0]) * (random_box[3] - random_box[1]) == (box[2] - box[0]) * (box[3] - box[1])


def test_proxy_roi_is_deterministic_and_normalized() -> None:
    arr = np.zeros((80, 100), dtype=np.uint8)
    arr[24:52, 58:86] = np.indices((28, 28)).sum(axis=0) % 2 * 255
    image = Image.fromarray(arr)
    first, diagnostics = proxy_gradient_energy_bbox(image, window_fraction=0.3, grid_size=11)
    second, _ = proxy_gradient_energy_bbox(image, window_fraction=0.3, grid_size=11)
    assert first == second
    assert 0 <= first[0] < first[2] <= 1
    assert 0 <= first[1] < first[3] <= 1
    assert diagnostics["content_fraction"] > 0


def test_built_pilot_has_no_patient_overlap_or_assistant_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    cohort_path = root / "results/pilot_v0/cohort_audit.csv"
    if not cohort_path.exists():
        return
    import pandas as pd

    cohort = pd.read_csv(cohort_path)
    assert not cohort.patient_id.duplicated().any()
    assert set(cohort.groupby("patient_id").split.nunique()) == {1}
    for split in ["train", "val", "test"]:
        path = root / f"results/pilot_v0/evidence_prompts/{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert all(message["role"] != "assistant" for message in row["messages"])
            assert "solution" not in row
            rendered = json.dumps(row["messages"], ensure_ascii=False)
            assert "pseudo_report" not in rendered
            assert "training_report" not in rendered
            assert all(Path(image).is_file() for image in row["images"])
