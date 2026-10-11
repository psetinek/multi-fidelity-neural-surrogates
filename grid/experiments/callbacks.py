"""Hydra callbacks for this project.

Enabled in `configs/config.yaml` so they apply to every multirun.
Each callback wraps its work in a try/except so a failing callback
never takes down the sweep itself — we log the failure and move on.
"""

import logging
import time
from pathlib import Path
from typing import Any

from hydra.core.utils import JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


log = logging.getLogger(__name__)


class AggregateResultsCallback(Callback):
    """On `on_multirun_end`, glob every `results/<axis>/*/result.csv`
    via `experiments._common.get_all_results` and write the snapshot to
    `results/_aggregate.csv` (overwritten each run).

    Gives us a point-in-time cumulative view without needing to call
    `get_all_results()` by hand — useful for plotting scripts that can
    just read one file.
    """

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            from experiments._common import get_all_results

            results_root = Path("results").resolve()
            if not results_root.exists():
                log.info("[aggregate] no results/ dir; skipping")
                return

            df = get_all_results(results_root)
            if df.empty:
                log.info("[aggregate] no result.csvs found; skipping")
                return

            out = results_root / "_aggregate.csv"
            df.to_csv(out, index=False)
            log.info(f"[aggregate] wrote {out} ({len(df)} rows)")
        except Exception as e:
            log.warning(f"[aggregate] failed: {e!r}")


class SweepSummaryCallback(Callback):
    """Accumulates per-job returns through `on_job_end`, then prints
    a compact summary on `on_multirun_end`: succeeded / failed counts,
    wallclock duration, sweep output dir.

    `LogJobReturnCallback` already logs per-job `return_value`; this
    callback adds the sweep-level header/footer.
    """

    def __init__(self) -> None:
        self._jobs: list[tuple[JobStatus, list[str]]] = []
        self._start: float | None = None

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._jobs = []
        self._start = time.time()
        log.info("[summary] multirun starting")

    def on_job_end(self, config: DictConfig, job_return: Any, **kwargs: Any) -> None:
        try:
            self._jobs.append((job_return.status, list(job_return.overrides)))
        except Exception as e:
            log.warning(f"[summary] on_job_end: couldn't record status: {e!r}")

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            duration = time.time() - (self._start or time.time())
            n_total = len(self._jobs)
            n_ok = sum(1 for status, _ in self._jobs if status == JobStatus.COMPLETED)
            n_fail = n_total - n_ok

            log.info("=== sweep summary ===")
            log.info(f"  jobs:      {n_ok}/{n_total} completed, {n_fail} failed")
            log.info(f"  duration:  {duration:.0f} s ({duration / 60:.1f} min)")
            log.info(f"  sweep dir: {config.hydra.sweep.dir}")

            if n_fail:
                log.info("  failed overrides:")
                for status, overrides in self._jobs:
                    if status != JobStatus.COMPLETED:
                        log.info(f"    - {' '.join(overrides)}")
        except Exception as e:
            log.warning(f"[summary] on_multirun_end failed: {e!r}")
