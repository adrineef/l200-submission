import datetime as dt
import os
from typing import Optional
from zoneinfo import ZoneInfo

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

_SEASON_BY_MONTH = {
    12: "hiver", 1: "hiver", 2: "hiver",
    3: "printemps", 4: "printemps", 5: "printemps",
    6: "été", 7: "été", 8: "été",
    9: "automne", 10: "automne", 11: "automne",
}
_SOUTH = {"hiver": "été", "été": "hiver", "printemps": "automne", "automne": "printemps"}


def compute_season(day: dt.date, hemisphere: str = "north") -> str:
    """Saison météorologique pour une date (hémisphère nord par défaut)."""
    season = _SEASON_BY_MONTH[day.month]
    return _SOUTH[season] if hemisphere == "south" else season


def inject_season_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """Écrit current_date et current_season dans l'état avant chaque tour."""
    tz = ZoneInfo(os.environ.get("HAIKU_TIMEZONE", "Europe/Paris"))
    today = dt.datetime.now(tz).date()
    callback_context.state["current_date"] = today.isoformat()
    callback_context.state["current_season"] = compute_season(
        today, os.environ.get("HAIKU_HEMISPHERE", "north")
    )
    return None
