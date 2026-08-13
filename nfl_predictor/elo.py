"""A self-contained NFL Elo model, FiveThirtyEight-style.

We walk every game in chronological order, record each team's pre-game rating
(so the features never peek at the result), then update both teams from the
final score with a margin-of-victory multiplier. Ratings carry across seasons
but regress a third of the way back to the 1500 mean each offseason, because
NFL rosters turn over hard.
"""

import numpy as np
import pandas as pd

MEAN = 1500.0
HOME_ADV = 55.0          # Elo points added to the home side
K = 20.0                 # base learning rate
SEASON_REGRESS = 1 / 3   # revert this fraction toward the mean each offseason


def expected_home(elo_home: float, elo_away: float, neutral: bool) -> float:
    """Logistic Elo win probability for the home team."""
    adv = 0.0 if neutral else HOME_ADV
    return 1.0 / (1.0 + 10 ** (-(elo_home + adv - elo_away) / 400.0))


def _mov_multiplier(margin: int, elo_diff: float) -> float:
    """Blow-outs move ratings more, but the autocorrelation correction damps
    the update when a big favorite wins big (538's formula)."""
    return np.log(abs(margin) + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2))


def run_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Given the schedule (chronological), return one row per game with the
    pre-game ratings and the Elo home win probability.

    Expects columns: game_id, season, gameday, gametime, home_team,
    away_team, home_score, away_score, location.
    """
    g = games.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}

    out = []
    for row in g.itertuples(index=False):
        home, away, season = row.home_team, row.away_team, int(row.season)

        for team in (home, away):
            if team not in ratings:
                ratings[team] = MEAN
                last_season[team] = season
            elif season > last_season[team]:
                # offseason regression toward the mean (handles byes/gaps too)
                ratings[team] += (MEAN - ratings[team]) * SEASON_REGRESS
                last_season[team] = season

        neutral = str(getattr(row, "location", "Home")).lower() == "neutral"
        elo_home, elo_away = ratings[home], ratings[away]
        p_home = expected_home(elo_home, elo_away, neutral)

        out.append({
            "game_id": row.game_id,
            "home_elo_pre": elo_home,
            "away_elo_pre": elo_away,
            "elo_diff": elo_home + (0.0 if neutral else HOME_ADV) - elo_away,
            "elo_prob_home": p_home,
        })

        # Update only from played games.
        hs, as_ = row.home_score, row.away_score
        if pd.notna(hs) and pd.notna(as_):
            result = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            margin = int(hs - as_)
            diff_winner = (elo_home - elo_away) if result == 1.0 else (elo_away - elo_home)
            mult = _mov_multiplier(margin if margin != 0 else 1, diff_winner)
            shift = K * mult * (result - p_home)
            ratings[home] += shift
            ratings[away] -= shift

    return pd.DataFrame(out)


def current_ratings(games: pd.DataFrame) -> dict[str, float]:
    """Replay all *played* games and return the latest rating per team, for
    projecting upcoming (unplayed) matchups."""
    played = games[games["home_score"].notna() & games["away_score"].notna()]
    g = played.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    for row in g.itertuples(index=False):
        home, away, season = row.home_team, row.away_team, int(row.season)
        for team in (home, away):
            if team not in ratings:
                ratings[team] = MEAN
                last_season[team] = season
            elif season > last_season[team]:
                ratings[team] += (MEAN - ratings[team]) * SEASON_REGRESS
                last_season[team] = season
        neutral = str(getattr(row, "location", "Home")).lower() == "neutral"
        p_home = expected_home(ratings[home], ratings[away], neutral)
        result = 1.0 if row.home_score > row.away_score else (
            0.0 if row.home_score < row.away_score else 0.5)
        margin = int(row.home_score - row.away_score)
        diff_winner = ((ratings[home] - ratings[away]) if result == 1.0
                       else (ratings[away] - ratings[home]))
        mult = _mov_multiplier(margin if margin != 0 else 1, diff_winner)
        shift = K * mult * (result - p_home)
        ratings[home] += shift
        ratings[away] -= shift
    return ratings
