#!/usr/bin/env bash
# Refresh nflverse data, retrain the ensemble, and print the picks board.
#   bash run.sh            -> next upcoming games
#   bash run.sh 1          -> just Week 1's slate
cd "$(dirname "$0")" || exit 1
[ -d .venv ] && source .venv/bin/activate

echo "==> Fetching schedules + play-by-play EPA (this takes a few minutes)..."
python3 -m nfl_predictor fetch
echo "==> Training Elo + logistic-regression ensemble..."
python3 -m nfl_predictor train
echo

if [ -n "${1:-}" ]; then
  echo "==> Picks for Week $1:"
  python3 -m nfl_predictor predict --week "$1"
else
  echo "==> Next upcoming picks:"
  python3 -m nfl_predictor predict --limit 16
fi
