"""Score upcoming games with the trained ensemble and render the picks board
in the same layout as the reference: WHEN · MATCHUP · SPREAD · WINNER · CONF."""

import numpy as np
import pandas as pd

from .config import team_name
from .features import feature_matrix, upcoming_frame


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ensemble win probabilities and a pick to a feature frame."""
    if frame.empty:
        return frame.assign(p_home_win=[], p_away_win=[], winner=[], confidence=[])

    X = feature_matrix(frame)
    p_lr = bundle["pipeline"].predict_proba(X)[:, 1]
    p_elo = frame["elo_prob_home"].to_numpy()
    w = bundle["blend_weight_lr"]
    p_home = np.clip(w * p_lr + (1 - w) * p_elo, 1e-6, 1 - 1e-6)

    out = frame.copy()
    out["p_home_win"] = p_home
    out["p_away_win"] = 1 - p_home
    home_pick = p_home >= 0.5
    out["winner_abbr"] = np.where(home_pick, out["home_team"], out["away_team"])
    out["winner"] = out["winner_abbr"].map(team_name)
    out["confidence"] = np.where(home_pick, p_home, 1 - p_home)
    return out


def _fmt_when(row) -> str:
    """'THU 7:00P' from weekday + gametime."""
    wd = str(row.get("weekday", "") or "")[:3].upper()
    t = str(row.get("gametime", "") or "13:00")
    try:
        hh, mm = t.split(":")[:2]
        hh = int(hh)
        ap = "A" if hh < 12 else "P"
        hh12 = hh % 12 or 12
        return f"{wd} {hh12}:{mm}{ap}"
    except Exception:
        return wd


def _fmt_spread(row) -> str:
    """'CIN -6.5': the favored side and the (home) spread line."""
    sp = row.get("spread_line", 0.0)
    try:
        sp = float(sp)
    except Exception:
        sp = 0.0
    if sp == 0.0:
        return "PK"
    # nflverse convention: spread_line > 0 means the home team is favored.
    fav = row["home_team"] if sp > 0 else row["away_team"]
    return f"{fav} {-abs(sp):.1f}"


def _bar(conf: float, width: int = 10) -> str:
    filled = int(round(conf * width))
    return "█" * filled + "·" * (width - filled)


def render_board(df: pd.DataFrame, limit: int | None = None) -> str:
    """A monospace picks board matching the reference layout."""
    if df.empty:
        return "No upcoming games to predict — run `fetch` first."
    df = df.sort_values(["gameday", "gametime", "game_id"])
    if limit:
        df = df.head(limit)

    head = f"{'WHEN':<10} {'MATCHUP':<14} {'SPREAD':<12} {'WINNER':<14} {'CONF':>6}"
    lines = [head, "-" * len(head)]
    for row in df.itertuples(index=False):
        r = row._asdict()
        when = _fmt_when(r)
        matchup = f"{r['away_team']} @ {r['home_team']}"
        spread = _fmt_spread(r)
        winner = r["winner"]
        conf = r["confidence"]
        lines.append(f"{when:<10} {matchup:<14} {spread:<12} {winner:<14} "
                     f"{conf*100:>5.1f}%  {_bar(conf)}")
    n = len(df)
    avg_spread = pd.to_numeric(df["spread_line"], errors="coerce").abs().mean()
    lines.append("-" * len(head))
    lines.append(f"{n} GAMES · AVG SPREAD {avg_spread:.1f} PTS")
    return "\n".join(lines)


def predict_upcoming(bundle: dict, limit=None, week=None, injuries=False) -> None:
    frame = upcoming_frame()
    if week is not None:
        frame = frame[frame["week"] == int(week)]
    preds = predict_frame(bundle, frame)
    print(render_board(preds, limit))
    if injuries:
        from .injuries import render_injuries
        shown = preds.sort_values(["gameday", "gametime", "game_id"])
        if limit:
            shown = shown.head(limit)
        print("\n" + render_injuries(shown))


def predict_matchup(bundle: dict, home: str, away: str) -> None:
    frame = upcoming_frame()
    m = frame[(frame["home_team"] == home) & (frame["away_team"] == away)]
    if m.empty:
        print(f"No scheduled {away} @ {home} game found in the upcoming slate.")
        return
    print(render_board(predict_frame(bundle, m)))
