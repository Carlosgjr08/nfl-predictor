"""Key injuries as board *context* — not a model feature.

We tested injury load as a 21st feature and it didn't improve accuracy (the
betting line already prices injuries in), so injuries never enter the model.
This module just surfaces who's Out/Doubtful for the games you're looking at,
so you can eyeball a pick before trusting it.
"""

import pandas as pd

from .config import INJURIES_CSV

# Rough importance order, so the most game-relevant names show first.
POS_RANK = {"QB": 0, "WR": 1, "CB": 2, "LT": 3, "T": 3, "OT": 3, "EDGE": 4,
            "DE": 4, "RB": 5, "TE": 6, "S": 7, "FS": 7, "SS": 7, "LB": 8,
            "OLB": 8, "ILB": 8, "MLB": 8, "DT": 9, "G": 10, "OG": 10, "C": 10}
MAX_PER_TEAM = 4


def load_injuries() -> pd.DataFrame:
    if INJURIES_CSV.exists():
        return pd.read_csv(INJURIES_CSV)
    return pd.DataFrame()


def _team_lines(inj: pd.DataFrame, season: int, week: int, team: str) -> list[str]:
    rows = inj[(inj["season"] == season) & (inj["week"] == week)
              & (inj["team"] == team)].copy()
    if rows.empty:
        return []
    rows["rank"] = rows["position"].map(POS_RANK).fillna(50)
    # Out first, then Doubtful, then Questionable; within that by position.
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    rows["srank"] = rows["report_status"].map(order).fillna(3)
    rows = rows.sort_values(["srank", "rank"]).head(MAX_PER_TEAM)
    tag = {"Out": "OUT", "Doubtful": "Dbt", "Questionable": "Q"}
    return [f"{r.position} {r.full_name} ({tag.get(r.report_status, r.report_status)})"
            for r in rows.itertuples()]


def render_injuries(games: pd.DataFrame) -> str:
    """A compact 'key injuries' section for the games in `games` (needs season,
    week, home_team, away_team)."""
    inj = load_injuries()
    if inj.empty:
        return ("Key injuries: none on file — run `python -m nfl_predictor "
                "fetch` to pull the latest reports.")

    out = ["Key injuries (OUT / Dbt / Q):"]
    any_shown = False
    for row in games.sort_values(["gameday", "gametime", "game_id"]).itertuples(index=False):
        r = row._asdict()
        season, week = int(r["season"]), int(r["week"])
        blocks = []
        for side in ("away_team", "home_team"):
            lines = _team_lines(inj, season, week, r[side])
            if lines:
                blocks.append(f"    {r[side]:<4} " + "; ".join(lines))
        if blocks:
            any_shown = True
            out.append(f"  {r['away_team']} @ {r['home_team']}")
            out.extend(blocks)
    if not any_shown:
        return "Key injuries: nothing notable reported for these games."
    return "\n".join(out)
