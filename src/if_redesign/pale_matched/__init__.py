"""Portable matched PALE heads; no clinical caches or campaign side effects."""
from .heads import VariantHead, make_head

P4_SPEC = dict(adapt="none", order="adapt_phi_mean", readout="laplace",
               site="attention", anchor=True, ranking=0.1, weighting="centre")

def make_p4_head(context_dim=0):
    """Return the actual P4 head; caller supplies source-fitted frame features."""
    return make_head(context_dim, dict(P4_SPEC))
