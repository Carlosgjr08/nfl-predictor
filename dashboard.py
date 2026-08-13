"""Browser dashboard for the NFL predictor — the WHEN / MATCHUP / SPREAD /
WINNER / CONF board, with confidence bars, mirroring the picks layout.

    pip install streamlit
    streamlit run dashboard_nfl.py
"""

import pandas as pd
import streamlit as st

from nfl_predictor.config import team_name
from nfl_predictor.features import upcoming_frame
from nfl_predictor.predict import _fmt_spread, _fmt_when, predict_frame
from nfl_predictor.train import load_bundle

st.set_page_config(page_title="NFL 2026 Predictor", page_icon="🏈", layout="wide")
st.title("🏈 NFL 2026 Predictor")
st.caption("Elo ratings + logistic regression over 20 features (EPA form, rest, "
           "venue, market lines) — trained on 23 seasons of nflverse data.")


@st.cache_resource
def _bundle():
    return load_bundle()


@st.cache_data(ttl=1800)
def _predictions() -> pd.DataFrame:
    return predict_frame(_bundle(), upcoming_frame())


df = _predictions()
if df.empty:
    st.info("No upcoming games — run `python -m nfl_predictor fetch` then `train`.")
    st.stop()

weeks = sorted(df["week"].dropna().unique())
week = st.selectbox("Week", weeks, format_func=lambda w: f"Week {int(w)}")
games = df[df["week"] == week].copy()
games = games.sort_values(["gameday", "gametime", "game_id"])

board = pd.DataFrame({
    "WHEN": [_fmt_when(r) for _, r in games.iterrows()],
    "MATCHUP": games["away_team"] + " @ " + games["home_team"],
    "SPREAD": [_fmt_spread(r) for _, r in games.iterrows()],
    "WINNER": games["winner"],
    "CONF": games["confidence"],
})
st.dataframe(
    board, hide_index=True, width="stretch",
    column_config={
        "CONF": st.column_config.ProgressColumn(
            "CONF", format="percent", min_value=0.0, max_value=1.0)},
)

n = len(games)
avg_spread = pd.to_numeric(games["spread_line"], errors="coerce").abs().mean()
st.caption(f"{n} GAMES · AVG SPREAD {avg_spread:.1f} PTS")

with st.expander("How the pick is made"):
    st.markdown(
        "- **Elo** rates every team, updated game-by-game with a "
        "margin-of-victory multiplier and offseason regression.\n"
        "- A **logistic regression** over 20 features (Elo, EPA form, rest, "
        "venue, weather, market lines) produces the win probability.\n"
        "- The two are **blended** at the weight that minimised log loss on a "
        "held-out season.")
