"""Track model predictions vs. actual results over the season.

Locks in the model's pick for each upcoming game *before* it's played — the
projected winner and how confident the ensemble was — then scores it once you
enter the final. Keeps a running log in data/nfl_predictions_log.csv and
regenerates the live scoreboard in RESULTS.md.

Each game week:
    python -m nfl_predictor track record                      # log the slate
    python -m nfl_predictor track result --home KC --away DEN --score 24-17
    python -m nfl_predictor track board                       # refresh scoreboard
"""

import json
from datetime import datetime

import pandas as pd

from .config import ROOT, PRED_LOG_CSV, MODELS_DIR, PREDICT_SEASON, team_name
from .features import build_frame, upcoming_frame
from .predict import _fmt_spread, _fmt_when, predict_frame

RESULTS_MD = ROOT / "RESULTS.md"

LOG_COLUMNS = [
    "game_id", "week", "when", "home_team", "away_team", "spread",
    "predicted_winner", "p_home_win", "confidence",
    "actual_home_score", "actual_away_score", "actual_winner", "hit",
]


def _is_blank(value) -> bool:
    """True for an un-filled cell, however pandas happens to store it."""
    return pd.isna(value) or str(value).strip() in ("", "nan", "<NA>", "None")


def _winner_abbr(home, away, hs, as_) -> str:
    if hs > as_:
        return home
    return away if hs < as_ else "Tie"


def load_log() -> pd.DataFrame:
    if PRED_LOG_CSV.exists():
        return pd.read_csv(PRED_LOG_CSV)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df: pd.DataFrame) -> None:
    PRED_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_LOG_CSV, index=False)


def record(bundle: dict) -> None:
    """Predict every upcoming game and append any not already logged
    (matched on game_id), with blank actuals to fill in later."""
    preds = predict_frame(bundle, upcoming_frame())
    if preds.empty:
        print("No upcoming games to log — run `fetch` first.")
        return
    log = load_log()
    existing = set(log["game_id"].astype(str))

    new_rows = []
    for row in preds.itertuples(index=False):
        r = row._asdict()
        if str(r["game_id"]) in existing:
            continue
        new_rows.append({
            "game_id": r["game_id"], "week": int(r["week"]),
            "when": _fmt_when(r), "home_team": r["home_team"],
            "away_team": r["away_team"], "spread": _fmt_spread(r),
            "predicted_winner": r["winner_abbr"],
            "p_home_win": round(float(r["p_home_win"]), 3),
            "confidence": round(float(r["confidence"]), 3),
            "actual_home_score": "", "actual_away_score": "",
            "actual_winner": "", "hit": "",
        })

    if not new_rows:
        print("No new games to log — everything scheduled is already tracked.")
        return
    log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    log = log.sort_values(["week", "game_id"]).reset_index(drop=True)
    save_log(log)
    build_results_md(log)
    print(f"Logged {len(new_rows)} new game(s) -> {PRED_LOG_CSV}")


def enter_result(home: str, away: str, score: str) -> None:
    """Fill in the final for a logged game and score the pick. `score` is
    'H-A' (home-away), e.g. 24-17."""
    try:
        hs, as_ = (int(x) for x in score.replace(":", "-").split("-"))
    except ValueError:
        raise SystemExit(f"Couldn't read score '{score}'. Use the form 24-17.")

    log = load_log()
    mask = (log["home_team"] == home) & (log["away_team"] == away) & \
           log["actual_home_score"].map(_is_blank)
    if not mask.any():
        raise SystemExit(f"No un-scored logged game for {away} @ {home}. "
                         "Run `track record` first, or check the abbreviations.")
    i = log[mask].index[0]

    for col in ("actual_home_score", "actual_away_score", "actual_winner", "hit"):
        log[col] = log[col].astype(object)

    winner = _winner_abbr(home, away, hs, as_)
    log.loc[i, "actual_home_score"] = hs
    log.loc[i, "actual_away_score"] = as_
    log.loc[i, "actual_winner"] = winner
    log.loc[i, "hit"] = int(winner == log.loc[i, "predicted_winner"])
    save_log(log)
    build_results_md(log)

    mark = "✅ correct" if int(log.loc[i, "hit"]) else "❌ missed"
    print(f"Recorded {away} {as_}-{hs} {home}. Model picked "
          f"'{team_name(log.loc[i, 'predicted_winner'])}' — {mark}.")


def grade(bundle: dict, week: int | None = None, season: int = PREDICT_SEASON) -> None:
    """Auto-score every *already played* game: reconstruct the model's pre-game
    pick from the data and compare it to the real final — no manual entry.

    Uses the same pre-game features training does (Elo before the game, EPA form
    shifted by one), so the pick is what the model *would* have called, with no
    peeking at the result.
    """
    df = build_frame()
    played = df[(df["home_win"].notna()) & (df["season"] == season)].copy()
    if week is not None:
        played = played[played["week"] == week]
    if played.empty:
        print(f"No finished {season}"
              f"{f' week {week}' if week else ''} games in the data yet — "
              "run `fetch` once the games have been played.")
        return

    preds = predict_frame(bundle, played)
    rows = []
    for row in preds.itertuples(index=False):
        r = row._asdict()
        hs, as_ = r["home_score"], r["away_score"]
        winner = _winner_abbr(r["home_team"], r["away_team"], hs, as_)
        rows.append({
            "game_id": r["game_id"], "week": int(r["week"]),
            "when": _fmt_when(r), "home_team": r["home_team"],
            "away_team": r["away_team"], "spread": _fmt_spread(r),
            "predicted_winner": r["winner_abbr"],
            "p_home_win": round(float(r["p_home_win"]), 3),
            "confidence": round(float(r["confidence"]), 3),
            "actual_home_score": int(hs), "actual_away_score": int(as_),
            "actual_winner": winner,
            "hit": int(winner == r["winner_abbr"]),
        })
    graded = pd.DataFrame(rows)

    # Replace any existing rows for these games, then add the freshly graded ones.
    log = load_log()
    log = log[~log["game_id"].astype(str).isin(graded["game_id"].astype(str))]
    log = pd.concat([log, graded], ignore_index=True).sort_values(
        ["week", "game_id"]).reset_index(drop=True)
    save_log(log)
    build_results_md(log)

    s = _summary(graded)
    print(f"Graded {len(graded)} game(s): {s['hits']}/{s['played']} correct "
          f"({s['acc']:.0%}). Scoreboard -> {RESULTS_MD}")


def _summary(log: pd.DataFrame) -> dict:
    done = log[~log["hit"].map(_is_blank)].copy()
    if done.empty:
        return {"played": 0}
    done["hit"] = done["hit"].astype(float).astype(int)
    return {"played": len(done), "hits": int(done["hit"].sum()),
            "acc": done["hit"].mean()}


def _backtest_line() -> str:
    """One-line summary of the held-out backtest, if the report is present."""
    report = MODELS_DIR / "eval_report.json"
    if not report.exists():
        return ""
    d = json.loads(report.read_text())
    if "accuracy_ensemble" not in d:
        return ""
    return (f"_Backtest (trained ≤{d['test_season'] - 1}, tested on "
            f"{d['test_season']}): {d['accuracy_ensemble']:.0%} accuracy, "
            f"{d['log_loss_ensemble']:.3f} log loss over "
            f"{d['test_games']} games._")


def build_results_md(log: pd.DataFrame | None = None) -> None:
    if log is None:
        log = load_log()
    s = _summary(log)

    lines = ["# 🏈 NFL 2026 — Model vs. Reality", ""]
    bt = _backtest_line()
    if bt:
        lines += [bt, ""]
    if s["played"]:
        lines += [f"**Season record:** {s['hits']} / {s['played']} correct "
                  f"({s['acc']:.0%})", ""]
    else:
        lines += ["_No games scored yet — picks are locked in and waiting._", ""]

    lines += [
        "| Week | When | Matchup | Spread | Pick | Conf. | Actual | Hit |",
        "|:----:|------|---------|:------:|------|:-----:|:------:|:---:|",
    ]
    for row in log.itertuples(index=False):
        r = row._asdict()
        if not _is_blank(r["actual_home_score"]):
            actual = (f"{int(float(r['actual_home_score']))}-"
                      f"{int(float(r['actual_away_score']))}")
            mark = "✅" if int(float(r["hit"])) else "❌"
        else:
            actual, mark = "⏳", "⏳"
        lines.append(
            f"| {r['week']} | {r['when']} | {r['away_team']} @ {r['home_team']} "
            f"| {r['spread']} | **{team_name(r['predicted_winner'])}** "
            f"| {float(r['confidence']):.0%} | {actual} | {mark} |")

    lines += ["", f"_Last updated {datetime.now():%Y-%m-%d}. "
              "Generated by `python -m nfl_predictor track board`._"]
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def board() -> None:
    log = load_log()
    build_results_md(log)
    s = _summary(log)
    if s["played"]:
        print(f"Record: {s['hits']}/{s['played']} ({s['acc']:.0%}). "
              f"Scoreboard -> {RESULTS_MD}")
    else:
        print(f"{len(log)} games logged, none scored yet. Scoreboard -> {RESULTS_MD}")
