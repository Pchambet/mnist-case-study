"""Command-line entry point: `uv run mnist-study <stage>`."""

from __future__ import annotations

import argparse

from mnist_study import data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mnist-study", description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("data", help="download and verify MNIST into data/raw/")
    train = sub.add_parser("train", help="train and predict every (model, seed) pair")
    train.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    train.add_argument("--max-epochs", type=int, default=12)
    sub.add_parser("analyze", help="compute results/ tables from cached predictions")
    sub.add_parser("figures", help="render docs/figures/*.png from results/")
    sub.add_parser("report", help="build the static report site/index.html")
    sub.add_parser("fixture", help="rebuild the small test fixture from data/raw/")
    args = parser.parse_args(argv)

    if args.stage == "data":
        print(data.download())
    elif args.stage == "train":
        from mnist_study.train import TrainConfig, run_all

        run_all(data.load_split(), args.seeds, TrainConfig(max_epochs=args.max_epochs))
    elif args.stage == "analyze":
        from mnist_study.analyze import analyze

        analyze()
    elif args.stage == "figures":
        from mnist_study.figures import render_all

        render_all()
    elif args.stage == "report":
        from mnist_study.report import build_report

        print(build_report())
    elif args.stage == "fixture":
        print(data.make_fixture())


if __name__ == "__main__":
    main()
