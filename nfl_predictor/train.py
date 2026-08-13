"""Train the combined model.

The core classifier is a logistic regression over the 20 features (Elo is one
of them, so the regression learns how much to trust Elo vs. EPA form, rest and
the market). We then blend its probability with the pure-Elo probability and
pick the blend weight that minimises log loss on a held-out season — this is
the explicit Elo + logistic-regression ensemble.

Evaluation is a time-based split: train on the earliest seasons, test on the
most recent completed one, so the score reflects true forecasting.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLS, MODELS_DIR
from .features import feature_matrix, training_frame

MODEL_PATH = MODELS_DIR / "nfl_model.joblib"
REPORT_PATH = MODELS_DIR / "eval_report.json"


def _new_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")),
    ])


def _best_blend(p_lr: np.ndarray, p_elo: np.ndarray, y: np.ndarray) -> float:
    """Weight w on the logistic model (1-w on Elo) minimising log loss."""
    best_w, best_ll = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 51):
        p = np.clip(w * p_lr + (1 - w) * p_elo, 1e-6, 1 - 1e-6)
        ll = log_loss(y, p, labels=[0, 1])
        if ll < best_ll:
            best_ll, best_w = ll, float(w)
    return round(best_w, 2)


def main(test_season: int = 2025) -> None:
    df = training_frame()
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    print(f"Training on {len(train)} games (< {test_season}), "
          f"testing on {len(test)} games in {test_season}")

    Xtr, ytr = feature_matrix(train), train["home_win"].astype(int)
    pipe = _new_pipeline().fit(Xtr, ytr)

    report = {"train_games": int(len(train)), "test_games": int(len(test)),
              "test_season": test_season, "n_features": len(FEATURE_COLS)}

    if len(test):
        Xte, yte = feature_matrix(test), test["home_win"].astype(int).to_numpy()
        p_lr = pipe.predict_proba(Xte)[:, 1]
        p_elo = test["elo_prob_home"].to_numpy()
        blend_w = _best_blend(pipe.predict_proba(Xtr)[:, 1],
                              train["elo_prob_home"].to_numpy(),
                              train["home_win"].astype(int).to_numpy())
        p_mix = np.clip(blend_w * p_lr + (1 - blend_w) * p_elo, 1e-6, 1 - 1e-6)

        base = float(ytr.mean())
        report.update({
            "blend_weight_lr": blend_w,
            "accuracy_logreg": round(accuracy_score(yte, p_lr > 0.5), 3),
            "accuracy_elo": round(accuracy_score(yte, p_elo > 0.5), 3),
            "accuracy_ensemble": round(accuracy_score(yte, p_mix > 0.5), 3),
            "log_loss_logreg": round(log_loss(yte, p_lr, labels=[0, 1]), 4),
            "log_loss_elo": round(log_loss(yte, p_elo, labels=[0, 1]), 4),
            "log_loss_ensemble": round(log_loss(yte, p_mix, labels=[0, 1]), 4),
            "brier_ensemble": round(brier_score_loss(yte, p_mix), 4),
            "baseline_accuracy_home": round(float((yte == 1).mean()), 3),
            "baseline_log_loss": round(log_loss(
                yte, np.full(len(yte), base), labels=[0, 1]), 4),
        })
        coefs = pipe.named_steps["lr"].coef_[0]
        report["feature_weights"] = {f: round(float(c), 3)
                                     for f, c in sorted(zip(FEATURE_COLS, coefs),
                                                        key=lambda t: -abs(t[1]))}

    # Refit on every completed game for live predictions.
    Xall, yall = feature_matrix(df), df["home_win"].astype(int)
    final = _new_pipeline().fit(Xall, yall)
    blend_w = report.get("blend_weight_lr",
                         _best_blend(final.predict_proba(Xall)[:, 1],
                                     df["elo_prob_home"].to_numpy(),
                                     yall.to_numpy()))
    bundle = {"pipeline": final, "blend_weight_lr": blend_w,
              "feature_cols": FEATURE_COLS, "trained_on_games": int(len(df))}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved model -> {MODEL_PATH}")


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model — run `python -m nfl_predictor train` first.")
    return joblib.load(MODEL_PATH)
