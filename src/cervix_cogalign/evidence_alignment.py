"""Leakage-controlled evidence supervision and adapter-only objectives."""
from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn.functional as F


def clinical_prompt(clinical: dict, sites: list[int]) -> str:
    # Explicit allowlist: no pathology, patient ID, center, site labels or gains.
    metadata = {k: clinical.get(k, "unknown") for k in ("HPV", "TCT", "age")}
    return (
        "Research OCT diagnostic evidence assessment. Clinical information: "
        + json.dumps(metadata, ensure_ascii=False)
        + ". Each image is a three-frame strip from one OCT scanner site. Image order: "
        + ", ".join(f"site {s}" for s in sites)
        + ". Return JSON with evidence (site, OCT_finding, importance, clinical_interpretation) "
        "and final_diagnosis (positive/negative for CIN2+). Importance is high, low or unassessed. "
        "Describe only supported observations; state when morphology is unverified. "
        "A site reread finding is not a histopathological CIN2+ diagnosis."
    )


def chain_target(sites, labels, gains, pathology):
    med = float(np.median(gains))
    evidence = []
    for site, label, gain in zip(sites, labels, gains):
        finding = {1: "Site reread positive; specific morphology unverified",
                   0: "Site reread negative; specific morphology unverified",
                   -1: "Morphology and site reread unavailable"}[int(label)]
        evidence.append(dict(site=int(site), OCT_finding=finding,
                             importance="high" if gain > med else "low",
                             clinical_interpretation="Weak site-level supervision; not a CIN2+ pathology label"))
    return json.dumps(dict(evidence=evidence, final_diagnosis="positive" if pathology else "negative"))


def pairwise_rank_loss(scores, gains, min_gap=0.001):
    """Bradley-Terry utility ordering using the same LM high/low log odds."""
    delta = gains[:, None] - gains[None, :]
    valid = torch.isfinite(delta) & (delta > min_gap)
    if not valid.any():
        return scores.sum() * 0
    return F.softplus(-(scores[:, None] - scores[None, :])[valid]).mean()


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    return -F.logsigmoid(beta * ((chosen - rejected) - (reference_chosen - reference_rejected))).mean()


def sequence_logp(logits, labels):
    labels = labels[:, 1:]
    valid = labels != -100
    values = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (values * valid).sum(-1)


def counterfactual_pair(sites, gains, site_labels, all_sites):
    """Pick source-supervised deletion and within-patient negative donor.

    Does NOT assign counterfactual pathology or require diagnosis reversal.
    """
    candidates = [i for i, s in enumerate(sites) if site_labels[s - 1] == 1 and gains[i] > 0]
    if not candidates:
        return None
    ix = max(candidates, key=lambda i: gains[i])
    removed = int(sites[ix])
    donors = [s for s in all_sites if site_labels[s - 1] == 0 and s not in sites]
    return {"removed": removed, "donor": min(donors, key=lambda s: abs(s-removed)) if donors else None}


def preference_text(removed, intervention):
    # Matched diagnosis language prevents preference from teaching blanket negatives.
    suffix = "The remaining evidence and clinical information must be reassessed; patient pathology is not changed by this edit."
    chosen = f"The original evidence at site {removed} is not available after {intervention}; I cannot cite it as observed support. " + suffix
    rejected = f"The original evidence at site {removed} is still available after {intervention}; I cite it as observed support. " + suffix
    return chosen, rejected


def validate_oof(index, membership):
    if index not in membership["gain_indices"]:
        raise ValueError("Patient is not held out for utility generation")
    if index in membership["fit_indices"] or index in membership["cal_indices"]:
        raise ValueError("OOF teacher leakage")


def language_lora_targets(model):
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    blocked = ("visual", "vision_tower", "vision_model", "multi_modal_projector", "mm_projector")
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
            and n.rsplit(".", 1)[-1] in suffixes and not any(b in n for b in blocked)]


def assert_lora_only(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n for n in names):
        raise ValueError("Only LoRA parameters may be trainable")
    if any(any(b in n for b in ("visual", "vision_tower", "vision_model", "projector")) for n in names):
        raise ValueError("Vision encoder/projector must stay frozen")
    return names
