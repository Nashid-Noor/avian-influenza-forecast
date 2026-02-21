"""CLI entry point: ``python -m src.run <command>``."""

from __future__ import annotations

import argparse
import sys

from src.utils.logging import get_logger

log = get_logger("src.run")


def cmd_preprocess() -> None:
    from src.ingest import auto_ingest
    from src.preprocess import run_preprocessing

    events, source = auto_ingest()
    log.info("Ingested %d events from source=%s", len(events), source)
    weekly = run_preprocessing(events)
    log.info("Preprocessing done — %d weekly rows.", len(weekly))


def cmd_train() -> None:
    import pandas as pd

    from src.config import MODEL_META_PATH, WEEKLY_OUTBREAKS_PATH
    from src.backtest import residual_std as compute_residual_std, walk_forward_backtest
    from src.train import save_model, train_lgbm
    from src.utils.io import load_csv, save_json

    weekly = load_csv(WEEKLY_OUTBREAKS_PATH, parse_dates=["week_start"])
    model, meta = train_lgbm(weekly)

    # Compute residual_std from a quick backtest for uncertainty
    bt = walk_forward_backtest(weekly, n_splits=8)
    if not bt.empty:
        meta["residual_std"] = compute_residual_std(bt, model="lgbm")
    else:
        meta["residual_std"] = 1.0

    save_model(model, meta)
    log.info("Training complete. Model saved.")


def cmd_backtest() -> None:
    from src.backtest import run_backtest
    from src.config import WEEKLY_OUTBREAKS_PATH
    from src.utils.io import load_csv

    weekly = load_csv(WEEKLY_OUTBREAKS_PATH, parse_dates=["week_start"])
    metrics = run_backtest(weekly)
    log.info("Best model: %s", metrics.get("best_model"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avian Influenza Forecast CLI")
    parser.add_argument(
        "command",
        choices=["preprocess", "train", "backtest"],
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    dispatch = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "backtest": cmd_backtest,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
