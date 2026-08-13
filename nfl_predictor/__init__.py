"""NFL game predictor: Elo ratings + logistic regression over EPA and
schedule features, trained on 23 seasons of nflverse data."""

__all__ = ["config", "elo", "features", "fetch", "train", "predict"]
