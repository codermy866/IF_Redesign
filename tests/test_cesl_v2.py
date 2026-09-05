from __future__ import annotations

import numpy as np

from cervix_cogalign.cesl import SLOT_IDS, SourceOnlyDonorPool, clinical_matrix


def _metadata():
    result = []
    for index in range(8):
        result.append(
            {
                "id": f"case-{index}",
                "patient_id": f"patient-{index}",
                "center": "source-a" if index < 4 else "source-b",
                "nuisance_stratum": ("35to44", "negative", True),
                "available": [True] * len(SLOT_IDS),
                # Deliberately present to ensure matching logic does not need it.
                "label": index % 2,
            }
        )
    return result


def test_donor_pool_is_deterministic_and_outcome_blind():
    metadata = _metadata()
    pool = SourceOnlyDonorPool(metadata, [0, 1, 2, 3, 4, 5])
    first = pool.edges([6, 7], 1, draws=3, q_variant="main_matched", seed=17)
    altered = _metadata()
    for row in altered:
        row["label"] = 1 - row["label"]
    second = SourceOnlyDonorPool(altered, [0, 1, 2, 3, 4, 5]).edges([6, 7], 1, draws=3, q_variant="main_matched", seed=17)
    assert [(edge.recipient_index, edge.donor_index, edge.matching_level) for edge in first] == [
        (edge.recipient_index, edge.donor_index, edge.matching_level) for edge in second
    ]
    assert all(edge.donor_index in {0, 1, 2, 3, 4, 5} for edge in first)


def test_clinical_scaling_uses_source_train_indices_only():
    baseline, scaling = clinical_matrix([30, 50, 1000], ["-", "16", "-"], ["NILM", "HSIL", "NILM"], [0, 1])
    changed, changed_scaling = clinical_matrix([30, 50, -1000], ["-", "16", "-"], ["NILM", "HSIL", "NILM"], [0, 1])
    assert scaling == changed_scaling
    assert np.array_equal(baseline[:2], changed[:2])
