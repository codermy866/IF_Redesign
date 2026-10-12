"""Native-Qwen adaptation and historical information-fusion reference components."""

from .cfv_jev import ResidualHead, encode_evidence, extract_interventions, fit_residual

from .fusion import FittedFusion, fit_nonnegative_fusion, masked_mean_site_probability
from .labels import SiteSupervision, build_site_supervision, build_site_targets, parse_positive_sites
from .modeling import ConvNeXtFeatureTail, PatientFusionHead
from .pcgrad import gradient_cosine, symmetric_pcgrad
from .sopi import (
    brier_contributions,
    estimate_sopi,
    log_contributions,
    posterior_movement,
    sopi_contributions,
    stagewise_sopi,
)
from .stagewise import PriorAnchoredUpdater, masked_mean_site_evidence
from .training import G2PCGradTrainer, StepStatistics

__all__ = [
    "ResidualHead",
    "encode_evidence",
    "extract_interventions",
    "fit_residual",
    "ConvNeXtFeatureTail",
    "FittedFusion",
    "G2PCGradTrainer",
    "PatientFusionHead",
    "PriorAnchoredUpdater",
    "SiteSupervision",
    "StepStatistics",
    "brier_contributions",
    "build_site_supervision",
    "build_site_targets",
    "estimate_sopi",
    "fit_nonnegative_fusion",
    "gradient_cosine",
    "log_contributions",
    "masked_mean_site_evidence",
    "masked_mean_site_probability",
    "parse_positive_sites",
    "posterior_movement",
    "sopi_contributions",
    "stagewise_sopi",
    "symmetric_pcgrad",
]
