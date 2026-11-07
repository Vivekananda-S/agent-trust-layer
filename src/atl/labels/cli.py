"""`atl-label` CLI: build the seed queue, open the labelling app, report progress."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from atl.labels.sample import (
    SeedConfig,
    build_seed_queue,
    load_queue,
    load_train_traces,
    save_queue,
)
from atl.labels.schema import load_labels

logger = logging.getLogger(__name__)

app = typer.Typer(help="Hand labelling: seed queue, labelling app, progress.", no_args_is_help=True)

ConfigOpt = Annotated[Path, typer.Option("--config", help="Seed config YAML.")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def seed(
    config: ConfigOpt = Path("configs/labels/seed_v1.yaml"),
    force: Annotated[bool, typer.Option(help="Overwrite an existing queue.")] = False,
) -> None:
    """Build the stratified seed queue from the train split."""
    cfg = SeedConfig.from_yaml(config)
    if cfg.queue_path.exists() and not force:
        raise typer.BadParameter(
            f"{cfg.queue_path} exists; the queue is fixed once labelling starts"
        )
    queue = build_seed_queue(load_train_traces(cfg), cfg.n, cfg.seed, cfg.max_per_task)
    save_queue(queue, cfg.queue_path)
    logger.info("Wrote %d queued traces to %s", len(queue), cfg.queue_path)
    for stratum, count in sorted(Counter(i.stratum for i in queue).items()):
        logger.info("  %-50s %d", stratum, count)


@app.command("app")
def run_app(config: ConfigOpt = Path("configs/labels/seed_v1.yaml")) -> None:
    """Open the Streamlit labelling app in the browser."""
    script = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--", "--config", str(config)]
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def stats(config: ConfigOpt = Path("configs/labels/seed_v1.yaml")) -> None:
    """Report labelling progress and class counts."""
    cfg = SeedConfig.from_yaml(config)
    queued = {i.trace_id for i in load_queue(cfg.queue_path)}
    labels = [lab for tid, lab in load_labels(cfg.labels_path).items() if tid in queued]
    verdicts = Counter(
        {True: "failed", False: "passed", None: "unjudgeable"}[x.failed] for x in labels
    )
    classes = Counter(c for x in labels for c in x.classes)
    others = sum(1 for x in labels if x.other_failure)
    logger.info("Labelled %d of %d queued traces: %s", len(labels), len(queued), dict(verdicts))
    logger.info("Classes: %s; other_failure: %d", dict(sorted(classes.items())), others)
