"""Convert OCT second-reading positions into masked site-level targets."""
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass


_NEGATIVE_TOKENS = {"", "0", "none", "negative", "normal", "nan", "无", "阴性", "未见"}


@dataclass(frozen=True)
class SiteSupervision:
    targets: tuple[int, ...]
    valid_mask: tuple[bool, ...]


def parse_positive_sites(value, number_of_sites=12):
    """Return sorted one-based positive OCT positions from a reread field."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(parse_positive_sites(item, number_of_sites))
        return tuple(sorted(set(values)))
    text = str(value).strip().lower()
    if text in _NEGATIVE_TOKENS:
        return tuple()
    positions = [int(token) for token in re.findall(r"\d+", text)]
    positions = [position for position in positions if position != 0]
    invalid = [position for position in positions if not 1 <= position <= number_of_sites]
    if invalid:
        raise ValueError(f"OCT positions outside 1..{number_of_sites}: {invalid}")
    return tuple(sorted(set(positions)))


def build_site_targets(value, number_of_sites=12):
    """Create one binary label for every OCT position."""
    positive = set(parse_positive_sites(value, number_of_sites))
    return [int(position in positive) for position in range(1, number_of_sites + 1)]


def build_site_supervision(value, available_sites, number_of_sites=12):
    """Mask unavailable sites instead of converting them to negatives."""
    positive = set(parse_positive_sites(value, number_of_sites))
    available = set(int(site) for site in available_sites)
    invalid = sorted(site for site in available if not 1 <= site <= number_of_sites)
    if invalid:
        raise ValueError(f"Available OCT positions outside 1..{number_of_sites}: {invalid}")
    missing_positive = sorted(positive - available)
    if missing_positive:
        raise ValueError(f"Positive reread positions are unavailable: {missing_positive}")
    return SiteSupervision(
        tuple(int(site in positive) for site in range(1, number_of_sites + 1)),
        tuple(site in available for site in range(1, number_of_sites + 1)),
    )
