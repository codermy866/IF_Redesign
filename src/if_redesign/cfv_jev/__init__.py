"""CFV-JEV core with an optional separately installed Visual-JEV backbone."""
from .evidence import LAPLACE_GRID, REPRESENTATIONS, encode_evidence
from .interventions import InterventionResult, extract_interventions, neutral_image
from .training import ResidualHead, center_balanced_nll, fit_residual
from .visual_jev import RawPatientReadout, UpstreamBindings, VisualJEVAdapter, prepare_patient
from .readout_training import FittedRawReadout, fit_raw_readout
from .backbone_training import (
    BackboneTrainingPolicy, BackboneTrainingResult, TrainingPatient,
    configure_language_lora, patient_answer_loss, source_selection_nll, train_source_backbone,
)

__all__ = [
    "FittedRawReadout", "fit_raw_readout", "UpstreamBindings", "prepare_patient", "source_selection_nll",
    "BackboneTrainingPolicy", "BackboneTrainingResult", "TrainingPatient",
    "configure_language_lora", "patient_answer_loss", "train_source_backbone",
    "InterventionResult", "LAPLACE_GRID", "REPRESENTATIONS", "RawPatientReadout",
    "ResidualHead", "VisualJEVAdapter", "center_balanced_nll", "encode_evidence",
    "extract_interventions", "fit_residual", "neutral_image",
]
