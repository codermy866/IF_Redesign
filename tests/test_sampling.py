from collections import Counter

from if_redesign.sampling import build_training_schedule


def test_supervision_matched_schedule_has_exact_exposure():
    rows = []
    for task in ("diag", "assim"):
        rows.extend({"task": task, "label": index % 2, "id": index} for index in range(20))
    rows.extend({"task": "site", "label": label, "id": index} for label in (0, 1) for index in range(20))
    schedule = build_training_schedule(
        rows, 120,
        {"diag": 0.2912430864, "site": 0.3532488749, "assim": 0.3555080387},
        site_positive_fraction=0.5,
    )
    counts = Counter(row["task"] for row in schedule)
    site = [row for row in schedule if row["task"] == "site"]
    assert counts == {"diag": 35, "site": 42, "assim": 43}
    assert sum(int(row["label"]) for row in site) == 21
