"""Core components for granularity-selective OCT phenotype fusion."""

from .fusion import FittedFusion, fit_nonnegative_fusion, masked_mean_site_probability
from .labels import SiteSupervision, build_site_supervision, build_site_targets, parse_positive_sites
from .modeling import ConvNeXtFeatureTail, PatientFusionHead
from .pcgrad import gradient_cosine, symmetric_pcgrad
from .training import G2PCGradTrainer, StepStatistics

__all__ = [
    "ConvNeXtFeatureTail", "FittedFusion", "G2PCGradTrainer", "PatientFusionHead",
    "SiteSupervision", "StepStatistics", "build_site_supervision", "build_site_targets",
    "fit_nonnegative_fusion", "gradient_cosine", "masked_mean_site_probability",
    "parse_positive_sites", "symmetric_pcgrad",
]
