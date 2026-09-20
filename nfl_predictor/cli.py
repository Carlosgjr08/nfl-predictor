"""Command-line entry point: fetch / train / predict."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nfl_predictor",
        description="Elo + logistic-regression NFL predictions over 23 seasons "
                    "of nflverse data with EPA form.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="pull schedules + play-by-play EPA")
    fetch.add_argument("--seasons", type=int, nargs="+", default=None)

    train = sub.add_parser("train", help="train + evaluate the ensemble")
    train.add_argument("--test-season", type=int, default=2025)

    predict = sub.add_parser("predict", help="predict the upcoming slate")
    predict.add_argument("--home", help="home team abbr (e.g. CIN)")
    predict.add_argument("--away", help="away team abbr (e.g. DET)")
    predict.add_argument("--week", type=int, help="only this week's games")
    predict.add_argument("--limit", type=int, default=None,
                         help="max games shown")
    predict.add_argument("--injuries", action="store_true",
                         help="also list key Out/Doubtful players per game")

    pre = sub.add_parser("preseason", help="exhibition board for preseason games "
                         "(never used for training)")
    pre.add_argument("--fetch", action="store_true",
                     help="pull the live slate from ESPN (needs an open network)")
    pre.add_argument("--year", type=int, default=None, help="preseason year")
    pre.add_argument("--limit", type=int, default=None, help="max games shown")

    track = sub.add_parser("track", help="log picks and score them vs. results")
    track_sub = track.add_subparsers(dest="track_command", required=True)
    track_sub.add_parser("record", help="log picks for the upcoming slate")
    result = track_sub.add_parser("result", help="enter a final score")
    result.add_argument("--home", required=True, help="home team abbr")
    result.add_argument("--away", required=True, help="away team abbr")
    result.add_argument("--score", required=True, help="final as H-A, e.g. 24-17")
    grade = track_sub.add_parser("grade", help="auto-score all played games "
                                 "against real results (no manual entry)")
    grade.add_argument("--week", type=int, help="only this week")
    track_sub.add_parser("board", help="refresh the live scoreboard")

    args = parser.parse_args()

    if args.command == "fetch":
        from .fetch import main as run
        run(args.seasons)
    elif args.command == "train":
        from .train import main as run
        run(args.test_season)
    elif args.command == "predict":
        from .predict import predict_matchup, predict_upcoming
        from .train import load_bundle
        bundle = load_bundle()
        if args.home and args.away:
            predict_matchup(bundle, args.home, args.away)
        else:
            predict_upcoming(bundle, args.limit, args.week, args.injuries)
    elif args.command == "preseason":
        from .preseason import run
        from .train import load_bundle
        from .config import PREDICT_SEASON
        run(load_bundle(), fetch=args.fetch,
            year=args.year or PREDICT_SEASON, limit=args.limit)
    elif args.command == "track":
        from . import track as tracker
        if args.track_command == "record":
            from .train import load_bundle
            tracker.record(load_bundle())
        elif args.track_command == "result":
            tracker.enter_result(args.home, args.away, args.score)
        elif args.track_command == "grade":
            from .train import load_bundle
            tracker.grade(load_bundle(), args.week)
        elif args.track_command == "board":
            tracker.board()


if __name__ == "__main__":
    main()
