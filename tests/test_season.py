import datetime as dt
from haiku_agent.season_tool import compute_season


def test_autumn_paris():
    assert compute_season(dt.date(2026, 10, 14)) == "automne"


def test_winter_boundary():
    assert compute_season(dt.date(2026, 12, 1)) == "hiver"
    assert compute_season(dt.date(2027, 2, 28)) == "hiver"


def test_southern_hemisphere():
    assert compute_season(dt.date(2026, 10, 14), "south") == "printemps"
