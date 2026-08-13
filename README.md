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
6. [Exiting and switching to another project (MLS / MLB)](#6-exiting-and-switching-to-another-project-mls--mlb)
7. [How to read the board](#7-how-to-read-the-board)
8. [How the model works](#8-how-the-model-works)
9. [The 20 features](#9-the-20-features)
10. [About preseason](#10-about-preseason)
11. [Troubleshooting](#11-troubleshooting)
12. [Project layout](#12-project-layout)

---

## 1. What you need first

- **Python 3.9 or newer.** Check with `python3 --version`.
- **An internet connection** (to download nflverse data the first time).
- That's it — no API keys, no accounts, no paid data.

## 2. Install

Open a terminal in this folder and run:

```bash
# 1. create an isolated environment (keeps this project separate)
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. install the dependencies
pip install -r requirements.txt
```

You only do this once. Every time you come back, just re-run
`source .venv/bin/activate` first.

> **Tip:** run the commands **one line at a time**, and skip the grey `#`
> comment lines — they're just notes for you, not commands. If your prompt ever
> changes to `quote>` or `dquote>` and seems stuck, press **`Control + C`** to
> cancel and get back to the normal prompt.

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
python -m nfl_predictor predict --week 1        # one week of games
python -m nfl_predictor predict --limit 16      # the next 16 upcoming games
python -m nfl_predictor predict --home CIN --away DET   # a single matchup
```
Prints the picks board. Team names are abbreviations (`CIN`, `KC`, `SF`…).

### `track` — keep score over the season
```bash
python -m nfl_predictor track record                        # log the upcoming picks
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
python -m nfl_predictor fetch          # pull the latest results
python -m nfl_predictor train          # retrain on the new data
python -m nfl_predictor track record   # log the upcoming picks
python -m nfl_predictor predict --week <N>   # see the picks
# ...after games finish, enter the finals:
python -m nfl_predictor track result --home KC --away DEN --score 24-17
python -m nfl_predictor track board    # update RESULTS.md
```

## 6. Exiting and switching to another project (MLS / MLB)

Each project (NFL, MLS, MLB) has its **own** `.venv` folder, so before you jump
to another one you should exit this project's environment cleanly. Two things
to know:

**Stop whatever is running.** If a command is still running — most often the
dashboard (`streamlit run dashboard.py`), which stays open on purpose — stop it
by pressing:

```
Ctrl + C          (hold Ctrl, press C)
```

That returns you to the normal prompt. (One-shot commands like `predict` or
`train` stop on their own when they finish — nothing to exit.)

**Leave this project's environment.** When you're done, deactivate the venv:

```bash
deactivate
```

Your prompt loses the `(.venv)` tag — that's how you know you're out. Now switch
to the other project and activate *its* environment:

```bash
cd ../your-mls-project        # or wherever the MLS / MLB project lives
source .venv/bin/activate     # activate that projects own environment
```

That's the whole switch. Quick reference:

| I want to… | Do this |
|------------|---------|
| Stop the dashboard / a stuck command | `Ctrl + C` |
| Leave this project's environment | `deactivate` |
| Am I in an environment? | Look for `(.venv)` at the start of your prompt |
| Switch to the MLS/MLB project | `cd ../<that-project>` then `source .venv/bin/activate` |
| Close the terminal entirely | `exit` (or just close the window) |

> You never have to "uninstall" anything to switch. Each project keeps its
> packages inside its own `.venv`, so they never clash — just `deactivate` one
> and `activate` the next.

## 7. How to read the board

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

## 8. How the model works

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

## 9. The 20 features

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

## 10. About preseason

**Preseason is never used to train the model — that rule is firm.** In
preseason the starters barely play, so the results are noise; training on them
would only make the model worse. And nflverse carries no preseason data anyway
(its schedule and play-by-play are regular-season + playoffs only).

But there's an **exhibition mode** so you can still click through a preseason
slate in the same board — handy for kicking the tires before the real season.
It reuses the already-trained model (it never feeds preseason back into
training) and, since the players who matter are resting, the picks lean almost
entirely on carried-over Elo and last season's form. **Treat them as a UI
rehearsal, not a forecast.**

```bash
# Option A — pull the live slate from ESPN (works on a normal home network):
python -m nfl_predictor preseason --fetch

# Option B — no network access to ESPN? Fill in the matchups yourself.
python -m nfl_predictor preseason          # creates data/preseason_games.csv
#   ...open that file, type in this week's games (away_team, home_team, date),
#   then run it again:
python -m nfl_predictor preseason
```

Output looks just like the regular board, with a `PRESEASON · EXHIBITION MODE`
banner on top. Nothing here touches `train`, `fetch`, or your season tracking.

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No trained model — run train first` | Run `python -m nfl_predictor train`. |
| `predict` shows no games | Run `fetch` first, or the season's schedule isn't posted yet. |
| EPA columns are all 0 for early-season games | Normal — teams have no current-season EPA until they've played; the model leans on Elo until then. |
| `fetch` fails with a 403 / Forbidden | Your network is blocking a data host. This project already routes around the one commonly-blocked host by using nflverse's GitHub mirror; if it still fails, try again on a different network. |
| `ModuleNotFoundError` | Activate the venv (`source .venv/bin/activate`) and re-run `pip install -r requirements.txt`. |

## 12. Project layout

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
