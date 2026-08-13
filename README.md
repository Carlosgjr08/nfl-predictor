# 🏈 NFL Predictor

Predicts NFL game winners by combining an **Elo rating system** with a
**logistic regression** trained on **23 seasons (2003–2025)** of
[nflverse](https://github.com/nflverse) data — including each team's **EPA
(expected points added)** form. It prints a clean picks board:

```
WHEN       MATCHUP        SPREAD       WINNER           CONF
------------------------------------------------------------
SUN 1:00P  CLE @ JAX      JAX -7.5     Jaguars         79.2%  ████████··
SUN 4:25P  ARI @ LAC      LAC -10.5    Chargers        84.8%  ████████··
MON 8:15P  DEN @ KC       KC -3.0      Chiefs          56.7%  ██████····
------------------------------------------------------------
16 GAMES · AVG SPREAD 4.0 PTS
```

---

## Table of contents

1. [What you need first](#1-what-you-need-first)
2. [Install](#2-install)
3. [Quick start](#3-quick-start-3-commands)
4. [The commands, one by one](#4-the-commands-one-by-one)
5. [Your weekly routine during the season](#5-your-weekly-routine-during-the-season)
6. [How to read the board](#6-how-to-read-the-board)
7. [How the model works](#7-how-the-model-works)
8. [The 20 features](#8-the-20-features)
9. [About preseason](#9-about-preseason)
10. [Troubleshooting](#10-troubleshooting)
11. [Project layout](#11-project-layout)

---

## 1. What you need first

- **Python 3.9 or newer.** Check with `python3 --version`.
- **An internet connection** (to download nflverse data the first time).
- That's it — no API keys, no accounts, no paid data.

## 2. Install

Open a terminal in this folder and run:

```bash
# 1. create an isolated environment so this project's packages stay separate
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. install the dependencies
pip install -r requirements.txt
```

You only do this once. Every time you come back, just re-run
`source .venv/bin/activate` first.

## 3. Quick start (3 commands)

```bash
python -m nfl_predictor fetch      # download 23 seasons of data (~3 min, once)
python -m nfl_predictor train      # train the model + print how good it is
python -m nfl_predictor predict --week 1     # show Week 1 picks
```

> The repo already ships with the data (`data/*.csv`), so you can skip straight
> to `train` and `predict` if you want. Run `fetch` when you want fresh results.

## 4. The commands, one by one

Everything is run as `python -m nfl_predictor <command>`.

### `fetch` — download the data
```bash
python -m nfl_predictor fetch
python -m nfl_predictor fetch --seasons 2020 2021 2022 2023 2024 2025 2026
```
Pulls two things from nflverse and saves them into `data/`:
- **Schedules** — every game, with spreads, moneylines, rest, venue, weather.
- **Play-by-play EPA** — each team's offensive and defensive efficiency per game.

Takes a few minutes the first time (it downloads one season at a time). Re-run
it weekly during the season to pull in the latest results.

### `train` — build the model
```bash
python -m nfl_predictor train
python -m nfl_predictor train --test-season 2024   # test on a different year
```
Trains the Elo + logistic-regression ensemble and prints an **evaluation
report**: how accurately it predicted a season it was *not* trained on, versus
Elo alone and versus a naive "always pick the home team" baseline. The trained
model is saved to `models/` (recreated any time you re-run `train`).

### `predict` — get the picks
```bash
python -m nfl_predictor predict --week 1        # one week's slate
python -m nfl_predictor predict --limit 16      # the next 16 upcoming games
python -m nfl_predictor predict --home CIN --away DET   # a single matchup
```
Prints the picks board. Team names are abbreviations (`CIN`, `KC`, `SF`…).

### `track` — keep score over the season
```bash
python -m nfl_predictor track record                        # log this week's picks
python -m nfl_predictor track result --home KC --away DEN --score 24-17
python -m nfl_predictor track board                         # refresh the scoreboard
```
- **record** locks in the model's picks *before* games are played.
- **result** enters a final score (home-away) and marks the pick right or wrong.
- **board** rewrites `RESULTS.md`, a running scoreboard showing your record.

### Dashboard (in a web browser)
```bash
streamlit run dashboard.py
```
Opens the same board in your browser, with a week selector and confidence bars.

### One-shot helper
```bash
bash run.sh          # fetch + train + show next picks
bash run.sh 1        # fetch + train + show Week 1
```

## 5. Your weekly routine during the season

Once real games start, do this once a week:

```bash
source .venv/bin/activate
python -m nfl_predictor fetch          # pull last week's results
python -m nfl_predictor train          # retrain on the new data
python -m nfl_predictor track record   # log this week's picks
python -m nfl_predictor predict --week <N>   # see the picks
# ...after games finish, enter the finals:
python -m nfl_predictor track result --home KC --away DEN --score 24-17
python -m nfl_predictor track board    # update RESULTS.md
```

## 6. How to read the board

| Column | Meaning |
|--------|---------|
| **WHEN** | Day + kickoff time, e.g. `SUN 1:00P` |
| **MATCHUP** | `AWAY @ HOME` (home team on the right) |
| **SPREAD** | The betting line: favored team and points, e.g. `KC -3.0` |
| **WINNER** | The team the model picks |
| **CONF** | The model's confidence in that pick, with a bar |

Confidence near 50% is a coin-flip; the bar fills up as the model gets more
sure. NFL is high-variance — even good models land around 65% accuracy over a
full season, so treat these as informed leans, not guarantees.

## 7. How the model works

Three pieces work together:

1. **Elo ratings.** Every team gets a rating that rises when they win and falls
   when they lose, adjusted for margin of victory and home field, and pulled
   back toward average each offseason. This captures overall team strength.
2. **EPA form.** From play-by-play, each team gets a rolling measure of how many
   expected points they add on offense and allow on defense — recent games
   weighted more. This captures *how* a team is playing right now.
3. **Logistic regression.** A model that takes all 20 features (Elo, EPA, rest,
   venue, weather, betting lines) and outputs a win probability. Its answer is
   then **blended with the pure Elo probability** at the mix that scored best on
   held-out data.

The result on the held-out 2025 season: **~65% accuracy** and clearly better
calibrated than Elo alone or a home-team baseline (~54%).

## 8. The 20 features

| Group | Features |
|-------|----------|
| Elo | `elo_diff`, `elo_prob_home` |
| Betting market | `spread_line`, `home_ml_prob` |
| Rest | `rest_diff`, `home_rest`, `away_rest` |
| Context | `div_game`, `week`, `neutral_site` |
| Venue / weather | `is_dome`, `is_turf`, `temp`, `wind` |
| **EPA form** | `home_off_epa`, `away_off_epa`, `home_def_epa`, `away_def_epa`, `off_epa_diff`, `def_epa_diff` |

EPA features are each team's form **going into** the game (shifted so a game
never sees its own result — no cheating).

## 9. About preseason

**This model does not predict preseason games, and it can't** — nflverse
carries no preseason data at all (both the schedule and the play-by-play are
regular-season + playoffs only).

That's the right call anyway: preseason results are basically noise for judging
real team strength (starters barely play), so training on them would only make
the model worse. Preseason is the time to get this pipeline set up and tested —
then the model goes to work once the regular season kicks off.

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No trained model — run train first` | Run `python -m nfl_predictor train`. |
| `predict` shows no games | Run `fetch` first, or the season's schedule isn't posted yet. |
| EPA columns are all 0 for early-season games | Normal — teams have no current-season EPA until they've played; the model leans on Elo until then. |
| `fetch` fails with a 403 / Forbidden | Your network is blocking a data host. This project already routes around the one commonly-blocked host by using nflverse's GitHub mirror; if it still fails, try again on a different network. |
| `ModuleNotFoundError` | Activate the venv (`source .venv/bin/activate`) and re-run `pip install -r requirements.txt`. |

## 11. Project layout

```
nfl-predictor/
├── README.md              ← you are here
├── requirements.txt       ← the packages to install
├── run.sh                 ← one-shot: fetch + train + predict
├── dashboard.py           ← browser version of the board
├── data/
│   ├── nfl_games.csv      ← schedules (shipped, refreshed by `fetch`)
│   └── nfl_team_epa.csv   ← per-game team EPA (shipped, refreshed by `fetch`)
└── nfl_predictor/
    ├── config.py          ← seasons, the 20 features, team names
    ├── fetch.py           ← downloads data from nflverse
    ├── elo.py             ← the Elo rating engine
    ├── features.py        ← builds the 20-feature table
    ├── train.py           ← trains + evaluates the model
    ├── predict.py         ← makes picks + renders the board
    ├── track.py           ← logs picks and scores them vs. reality
    └── cli.py             ← ties the commands together
```

`models/` and `RESULTS.md` are created when you run `train` and `track`.
