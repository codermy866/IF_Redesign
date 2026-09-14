import pytest

from if_redesign.labels import build_site_targets, parse_positive_sites


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
