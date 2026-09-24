import pytest

from if_redesign.labels import build_site_supervision, build_site_targets, parse_positive_sites


def test_second_reading_positions_are_direct_site_targets():
    assert parse_positive_sites("2，5; 9") == (2, 5, 9)
    targets = build_site_targets("2,5,9", number_of_sites=12)
    assert sum(targets) == 3
    assert [index + 1 for index, value in enumerate(targets) if value] == [2, 5, 9]


def test_negative_and_invalid_second_readings():
    assert parse_positive_sites("阴性") == ()
    assert parse_positive_sites(0) == ()
    with pytest.raises(ValueError):
        parse_positive_sites("13", number_of_sites=12)


def test_unavailable_sites_are_masked_instead_of_becoming_negatives():
    supervision = build_site_supervision("2, 5", available_sites=(1, 2, 4, 5), number_of_sites=6)
    assert supervision.targets == (0, 1, 0, 0, 1, 0)
    assert supervision.valid_mask == (True, True, False, True, True, False)
    with pytest.raises(ValueError):
        build_site_supervision("3", available_sites=(1, 2), number_of_sites=6)
