"""Pull the two datasets we need from nflverse:

  1. the schedule (all 34 columns) — mirrored on GitHub because the primary
     habitatring host is blocked by the egress policy here.
  2. per-game, per-team EPA — aggregated from play-by-play, one season at a
     time so we never hold 23 seasons of raw plays in memory or on disk.

Run:  python -m nfl_predictor fetch  [--seasons 2003 ... 2026]
"""

import pandas as pd

from .config import (ALL_SEASONS, DATA_DIR, GAMES_CSV, INJURIES_CSV,
                     SCHEDULE_COLUMNS, SCHEDULE_URL, TEAM_EPA_CSV)


def fetch_schedules() -> pd.DataFrame:
    """The full nflverse game log, kept to the columns we care about."""
    games = pd.read_csv(SCHEDULE_URL, low_memory=False)
    cols = [c for c in SCHEDULE_COLUMNS if c in games.columns]
    return games[cols].copy()


def _team_epa_for_season(season: int) -> pd.DataFrame:
    """Offensive & defensive EPA per team per game for one season.

    A team's defensive EPA in a game is exactly its opponent's offensive EPA
    in that same game, so we build the offensive table then self-join on the
    opponent to attach the defensive side.
    """
    import nfl_data_py as nfl

    pbp = nfl.import_pbp_data([season], downcast=True, cache=False)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])
                & pbp["epa"].notna()
                & pbp["posteam"].notna()].copy()

    off = (plays.groupby(["game_id", "posteam", "defteam"])
                .agg(off_epa=("epa", "sum"), off_plays=("epa", "size"))
                .reset_index()
                .rename(columns={"posteam": "team", "defteam": "opponent"}))
    off["off_epa_per_play"] = off["off_epa"] / off["off_plays"]

    # Defence = the opponent's offence in the same game.
    defside = off.rename(columns={
        "team": "opponent", "opponent": "team",
        "off_epa": "def_epa", "off_plays": "def_plays",
        "off_epa_per_play": "def_epa_per_play"})
    merged = off.merge(defside, on=["game_id", "team", "opponent"], how="inner")
    merged["season"] = season
    return merged


def fetch_team_epa(seasons) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        print(f"  play-by-play EPA for {yr} ...", flush=True)
        try:
            frames.append(_team_epa_for_season(yr))
        except Exception as exc:  # a season with no PBP yet (e.g. future)
            print(f"    skipped {yr}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_injuries(seasons) -> pd.DataFrame:
    """Weekly injury reports (2009+). Shown as board context — not a model
    feature, since the betting line already prices injuries in."""
    import nfl_data_py as nfl

    yrs = [y for y in seasons if y >= 2009]   # reports don't exist before 2009
    if not yrs:
        return pd.DataFrame()
    try:
        inj = nfl.import_injuries(yrs)
    except Exception as exc:
        print(f"    (injuries unavailable: {exc})")
        return pd.DataFrame()
    keep = ["season", "week", "team", "position", "full_name", "report_status"]
    inj = inj[[c for c in keep if c in inj.columns]]
    return inj[inj["report_status"].isin(["Out", "Doubtful", "Questionable"])].copy()


def main(seasons=None) -> None:
    seasons = seasons or ALL_SEASONS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("==> Fetching schedules (GitHub mirror)...")
    games = fetch_schedules()
    games = games[games["season"].isin(seasons)].reset_index(drop=True)
    games.to_csv(GAMES_CSV, index=False)
    print(f"    {len(games)} games -> {GAMES_CSV}")

    print("==> Fetching play-by-play EPA (one season at a time)...")
    epa = fetch_team_epa(seasons)
    epa.to_csv(TEAM_EPA_CSV, index=False)
    print(f"    {len(epa)} team-games -> {TEAM_EPA_CSV}")

    print("==> Fetching injury reports (context)...")
    inj = fetch_injuries(seasons)
    inj.to_csv(INJURIES_CSV, index=False)
    print(f"    {len(inj)} injury rows -> {INJURIES_CSV}")
