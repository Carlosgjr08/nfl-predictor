"""Turn the raw schedule + per-game EPA into the modelling frame: one row per
game carrying the 20 features, plus the home-win label for played games.

The two rules that keep this honest:
  * Elo ratings are the *pre-game* values from elo.run_elo.
  * EPA features are each team's form going *into* the game — an exponentially
    weighted average of their previous games, shifted by one so a game never
    sees its own EPA.
"""

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, GAMES_CSV, PREDICT_SEASON, TEAM_EPA_CSV
from .elo import current_ratings, expected_home, run_elo

EPA_HALFLIFE = 10        # games; ~two thirds of a season of memory


def _load_games() -> pd.DataFrame:
    g = pd.read_csv(GAMES_CSV, low_memory=False)
    g = g[g["game_type"].notna()].copy()
    # a stable chronological key
    g["gametime"] = g["gametime"].fillna("13:00")
    return g


def _rolling_epa(epa: pd.DataFrame) -> pd.DataFrame:
    """Per team, the EWMA of offensive & defensive EPA/play *before* each game."""
    epa = epa.sort_values(["team", "season", "game_id"]).reset_index(drop=True)

    def _ewm_prev(s: pd.Series) -> pd.Series:
        # mean of prior games only: shift first, then ewm
        return s.shift(1).ewm(halflife=EPA_HALFLIFE, min_periods=1).mean()

    epa["off_form"] = epa.groupby("team")["off_epa_per_play"].transform(_ewm_prev)
    epa["def_form"] = epa.groupby("team")["def_epa_per_play"].transform(_ewm_prev)
    return epa[["game_id", "team", "off_form", "def_form"]]


def _implied_prob(moneyline: pd.Series) -> pd.Series:
    """American moneyline -> implied win probability (no vig removal)."""
    ml = pd.to_numeric(moneyline, errors="coerce")
    pos = 100.0 / (ml + 100.0)
    neg = (-ml) / (-ml + 100.0)
    return np.where(ml < 0, neg, pos)


def build_frame() -> pd.DataFrame:
    games = _load_games()
    elo = run_elo(games)
    form = _rolling_epa(pd.read_csv(TEAM_EPA_CSV))

    df = games.merge(elo, on="game_id", how="left")
    df = df.merge(form.rename(columns={"team": "home_team",
                                       "off_form": "home_off_epa",
                                       "def_form": "home_def_epa"}),
                  on=["game_id", "home_team"], how="left")
    df = df.merge(form.rename(columns={"team": "away_team",
                                       "off_form": "away_off_epa",
                                       "def_form": "away_def_epa"}),
                  on=["game_id", "away_team"], how="left")

    # --- assemble the 20 features -------------------------------------------
    df["home_ml_prob"] = _implied_prob(df["home_moneyline"])
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(float)
    df["is_turf"] = (~df["surface"].astype(str).str.contains("grass", case=False,
                                                             na=False)).astype(float)
    df["neutral_site"] = (df["location"].astype(str).str.lower() == "neutral").astype(float)
    df["off_epa_diff"] = df["home_off_epa"] - df["away_off_epa"]
    df["def_epa_diff"] = df["home_def_epa"] - df["away_def_epa"]

    # sensible fills: EPA form of a team's first-ever game = league average (0);
    # dome games have no wind/temp; missing market lines -> neutral.
    for col in ["home_off_epa", "away_off_epa", "home_def_epa", "away_def_epa",
                "off_epa_diff", "def_epa_diff"]:
        df[col] = df[col].fillna(0.0)
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce").fillna(
        pd.Series(np.where(df["is_dome"] == 1, 68.0, 60.0), index=df.index))
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)
    df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce").fillna(0.0)
    df["home_ml_prob"] = df["home_ml_prob"].fillna(df["elo_prob_home"])
    df["home_rest"] = pd.to_numeric(df["home_rest"], errors="coerce").fillna(7.0)
    df["away_rest"] = pd.to_numeric(df["away_rest"], errors="coerce").fillna(7.0)
    df["rest_diff"] = df["rest_diff"].fillna(0.0)
    df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce").fillna(0.0)

    df["home_win"] = np.where(df["home_score"] > df["away_score"], 1.0,
                       np.where(df["home_score"] < df["away_score"], 0.0, np.nan))
    return df


def training_frame() -> pd.DataFrame:
    """Only decided, regular+post-season games with a label."""
    df = build_frame()
    played = df[df["home_win"].notna()].copy()
    return played.reset_index(drop=True)


def upcoming_frame() -> pd.DataFrame:
    """Unplayed games (the slate to predict), with Elo projected from current
    ratings and EPA form carried from each team's most recent games."""
    df = build_frame()
    # Only the season we're predicting — a handful of old seasons have games
    # with no recorded score (data gaps), which must not leak into the slate.
    up = df[(df["home_win"].isna()) & (df["season"] == PREDICT_SEASON)].copy()

    # Project Elo for teams using their latest post-replay ratings.
    ratings = current_ratings(_load_games())
    mean = 1500.0
    up["home_elo_pre"] = up["home_team"].map(ratings).fillna(mean)
    up["away_elo_pre"] = up["away_team"].map(ratings).fillna(mean)
    neutral = up["neutral_site"] == 1
    from .elo import HOME_ADV
    up["elo_diff"] = up["home_elo_pre"] + np.where(neutral, 0.0, HOME_ADV) - up["away_elo_pre"]
    up["elo_prob_home"] = [
        expected_home(h, a, n)
        for h, a, n in zip(up["home_elo_pre"], up["away_elo_pre"], neutral)]
    up["home_ml_prob"] = up["home_ml_prob"].fillna(up["elo_prob_home"])
    up["off_epa_diff"] = up["home_off_epa"] - up["away_off_epa"]
    up["def_epa_diff"] = up["home_def_epa"] - up["away_def_epa"]
    return up.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].astype(float)
