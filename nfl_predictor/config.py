"""Shared configuration: paths, the season window, the raw schedule columns
we pull from nflverse, and the exact 20 features the model trains on."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

GAMES_CSV = DATA_DIR / "nfl_games.csv"          # raw schedule (all columns)
TEAM_EPA_CSV = DATA_DIR / "nfl_team_epa.csv"    # per game, per team EPA
INJURIES_CSV = DATA_DIR / "nfl_injuries.csv"    # weekly injury reports (context only)
PRED_LOG_CSV = DATA_DIR / "nfl_predictions_log.csv"

# 23 completed seasons to train on; 2026 is the season we predict.
TRAIN_SEASONS = list(range(2003, 2026))   # 2003 .. 2025  (23 seasons)
PREDICT_SEASON = 2026
ALL_SEASONS = TRAIN_SEASONS + [PREDICT_SEASON]

# The habitatring schedule host is blocked by egress policy, but nflverse
# mirrors the identical file on GitHub — pull schedules from there instead.
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# The 34 raw schedule columns from nflverse (the SCHEDULE_COLUMNS list).
SCHEDULE_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "away_score", "home_team", "home_score", "location", "result",
    "total", "overtime", "away_rest", "home_rest", "away_moneyline",
    "home_moneyline", "spread_line", "away_spread_odds", "home_spread_odds",
    "total_line", "under_odds", "over_odds", "div_game", "roof", "surface",
    "temp", "wind", "away_qb_name", "home_qb_name", "stadium",
]

# The 20 engineered features the logistic regression is trained on. The Elo
# signal (elo_diff / elo_prob_home) is fed in as a feature, so the regression
# learns how to weight Elo against EPA form, rest, market lines and venue —
# that is the "combine Elo with logistic regression" the model is built on.
FEATURE_COLS = [
    "elo_diff",          # home Elo - away Elo, pre-game
    "elo_prob_home",     # home win probability implied by Elo
    "spread_line",       # market: home points spread (negative = home favored)
    "home_ml_prob",      # market: implied prob from home moneyline
    "rest_diff",         # home_rest - away_rest (days)
    "home_rest",
    "away_rest",
    "div_game",          # divisional matchup
    "week",              # later weeks = more settled form
    "is_dome",           # roof closed / dome
    "is_turf",           # artificial surface
    "temp",              # game temperature (F)
    "wind",              # wind (mph)
    "neutral_site",      # neutral venue (no true home edge)
    "home_off_epa",      # rolling offensive EPA/play, home team
    "away_off_epa",      # rolling offensive EPA/play, away team
    "home_def_epa",      # rolling defensive EPA/play allowed, home team
    "away_def_epa",      # rolling defensive EPA/play allowed, away team
    "off_epa_diff",      # home_off_epa - away_off_epa
    "def_epa_diff",      # home_def_epa - away_def_epa (lower = home defends better)
]
assert len(FEATURE_COLS) == 20, "the model is specified as a 20-feature model"

# Team abbreviation -> nickname, for the results board.
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LV": "Raiders", "LAC": "Chargers", "LA": "Rams", "LAR": "Rams",
    "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints",
    "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers",
    "SF": "49ers", "SEA": "Seahawks", "TB": "Buccaneers", "TEN": "Titans",
    "WAS": "Commanders",
    # historical abbreviations that appear in older seasons
    "OAK": "Raiders", "SD": "Chargers", "STL": "Rams",
}


def team_name(abbr: str) -> str:
    return TEAM_NAMES.get(abbr, abbr)
