"""Preseason **exhibition** mode.

A sandbox for clicking through a preseason slate in the same board layout,
*without* letting preseason near the model. Two things make this safe and
honest:

  * It only ever *reads* the already-trained model — preseason games are never
    added to training (starters rest, so their results are noise).
  * nflverse carries no preseason data, so the schedule comes from either ESPN
    (live, when you run it on your own machine) or a small CSV you fill in.

Because the real players barely take the field, these picks lean almost
entirely on carried-over Elo and last season's form — treat them as a UI
rehearsal, not a forecast.

    python -m nfl_predictor preseason --fetch     # pull the live ESPN slate
    python -m nfl_predictor preseason             # use data/preseason_games.csv
"""

import json
import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_DIR, PREDICT_SEASON, team_name
from .elo import HOME_ADV, current_ratings, expected_home
from .features import EPA_HALFLIFE, _load_games
from .predict import predict_frame, render_board

PRESEASON_CSV = DATA_DIR / "preseason_games.csv"

# ESPN's public scoreboard. seasontype=1 is preseason. Reachable from a normal
# network (e.g. your Mac); some locked-down environments block it.
ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
            "scoreboard?limit=1000&seasontype=1&dates={year}")

# ESPN abbreviations that differ from the nflverse ones this project uses.
ESPN_TEAM_FIX = {"WSH": "WAS", "LAR": "LA"}


def _latest_epa_form() -> pd.DataFrame:
    """Each team's most recent EPA form (EWMA through their last game)."""
    epa = pd.read_csv(DATA_DIR / "nfl_team_epa.csv")
    epa = epa.sort_values(["team", "season", "game_id"])
    off = epa.groupby("team")["off_epa_per_play"].apply(
        lambda s: s.ewm(halflife=EPA_HALFLIFE).mean().iloc[-1])
    dff = epa.groupby("team")["def_epa_per_play"].apply(
        lambda s: s.ewm(halflife=EPA_HALFLIFE).mean().iloc[-1])
    return pd.DataFrame({"off_form": off, "def_form": dff})


def fetch_espn_preseason(year: int = PREDICT_SEASON, raw: dict | None = None) -> pd.DataFrame:
    """Live preseason schedule from ESPN as a tidy games frame. Pass `raw` (a
    parsed JSON dict) to parse without a network call (used in tests)."""
    if raw is None:
        req = urllib.request.Request(ESPN_URL.format(year=year),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            raw = json.load(resp)

    rows = []
    for ev in raw.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue

        def _abbr(side):
            a = teams[side].get("team", {}).get("abbreviation", "")
            return ESPN_TEAM_FIX.get(a, a)

        # ESPN gives the line as "KC -3.5"; positive spread_line = home favored.
        spread = 0.0
        odds = comp.get("odds") or []
        if odds:
            details = str(odds[0].get("details", "")).strip()
            fav_abbr, _, num = details.rpartition(" ")
            try:
                pts = float(num)
                home = _abbr("home")
                spread = -pts if ESPN_TEAM_FIX.get(fav_abbr, fav_abbr) == home else pts
            except ValueError:
                spread = 0.0

        # ESPN timestamps are UTC; show them in US Eastern (the league's clock).
        dt = pd.to_datetime(ev.get("date"), errors="coerce", utc=True)
        if pd.notna(dt):
            try:
                dt = dt.tz_convert("America/New_York")
            except Exception:
                pass
        rows.append({
            "game_id": ev.get("id", f"{_abbr('away')}_{_abbr('home')}"),
            "gameday": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "",
            "gametime": dt.strftime("%H:%M") if pd.notna(dt) else "13:00",
            "weekday": dt.strftime("%A") if pd.notna(dt) else "",
            "away_team": _abbr("away"), "home_team": _abbr("home"),
            "spread_line": spread,
            "week": int((ev.get("week") or {}).get("number", 0) or 0),
        })
    return pd.DataFrame(rows)


def _write_template() -> None:
    """Drop a fill-in-yourself schedule so the command works with no network."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"gameday": "2026-08-07", "gametime": "19:00", "weekday": "Thursday",
         "away_team": "DET", "home_team": "CIN", "spread_line": 0.0, "week": 1},
        {"gameday": "2026-08-08", "gametime": "13:00", "weekday": "Friday",
         "away_team": "GB", "home_team": "PIT", "spread_line": 0.0, "week": 1},
    ]).to_csv(PRESEASON_CSV, index=False)


def preseason_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Attach the 20 model features to a set of preseason games, using Elo and
    EPA form carried over from the regular season (no preseason data exists)."""
    ratings = current_ratings(_load_games())
    form = _latest_epa_form()
    mean = 1500.0

    g = games.copy()
    if "game_id" not in g or g["game_id"].isna().any():
        g["game_id"] = (g.get("gameday", "").astype(str) + "_"
                        + g["away_team"].astype(str) + "_" + g["home_team"].astype(str))
    for col, default in (("gameday", ""), ("gametime", "13:00"), ("weekday", "")):
        if col not in g:
            g[col] = default
    g["home_elo"] = g["home_team"].map(ratings).fillna(mean)
    g["away_elo"] = g["away_team"].map(ratings).fillna(mean)
    g["elo_diff"] = g["home_elo"] + HOME_ADV - g["away_elo"]
    g["elo_prob_home"] = [expected_home(h, a, False)
                          for h, a in zip(g["home_elo"], g["away_elo"])]

    for side in ("home", "away"):
        g[f"{side}_off_epa"] = g[f"{side}_team"].map(form["off_form"]).fillna(0.0)
        g[f"{side}_def_epa"] = g[f"{side}_team"].map(form["def_form"]).fillna(0.0)
    g["off_epa_diff"] = g["home_off_epa"] - g["away_off_epa"]
    g["def_epa_diff"] = g["home_def_epa"] - g["away_def_epa"]

    # Unknown / not-applicable in preseason -> neutral defaults.
    g["spread_line"] = pd.to_numeric(g.get("spread_line", 0.0), errors="coerce").fillna(0.0)
    g["home_ml_prob"] = g["elo_prob_home"]
    g["home_rest"] = 7.0
    g["away_rest"] = 7.0
    g["rest_diff"] = 0.0
    g["div_game"] = 0.0
    g["week"] = pd.to_numeric(g.get("week", 0), errors="coerce").fillna(0.0)
    g["is_dome"] = 0.0
    g["is_turf"] = 0.0
    g["temp"] = 65.0
    g["wind"] = 0.0
    g["neutral_site"] = 0.0
    return g


def run(bundle: dict, fetch: bool = False, year: int = PREDICT_SEASON,
        limit=None) -> None:
    if fetch:
        try:
            games = fetch_espn_preseason(year)
        except Exception as exc:
            print(f"Couldn't reach ESPN ({exc}).\nIf you're on a restricted "
                  f"network, fill in {PRESEASON_CSV} by hand and run without "
                  "--fetch.")
            return
        if games.empty:
            print("ESPN returned no preseason games for that year.")
            return
        games.to_csv(PRESEASON_CSV, index=False)
        print(f"Pulled {len(games)} preseason games -> {PRESEASON_CSV}\n")
    else:
        if not PRESEASON_CSV.exists():
            _write_template()
            print(f"No schedule yet — I created a template at {PRESEASON_CSV}.\n"
                  "Fill in this week's preseason matchups (or run with --fetch "
                  "on an open network) and run again.")
            return
        games = pd.read_csv(PRESEASON_CSV)

    preds = predict_frame(bundle, preseason_frame(games))
    print("PRESEASON · EXHIBITION MODE")
    print("Picks lean on carried-over Elo + last season's form — starters rest "
          "in preseason, so treat these as a rehearsal, not a forecast.\n")
    print(render_board(preds, limit))
