"""Same-session alternating comparison protocol for the CUDA line.

This file contains no dataset loader or output-directory logic.
It captures the measurement boundary used by the paper: resident indexes are
loaded once, baseline and MODULI arms alternate in one process, every arm pays
the same launch, synchronization, and result-copy boundary, and statistics
are computed from the paired rounds with the standard-library definitions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Sequence


@dataclass(frozen=True)
class ArmResult:
    query_count: int
    wall_seconds: Sequence[float]
    recall_at_k: float

    @property
    def qps(self) -> List[float]:
        return [self.query_count / seconds for seconds in self.wall_seconds]

    @property
    def median_qps(self) -> float:
        return statistics.median(self.qps)

    @property
    def mean_qps(self) -> float:
        return statistics.mean(self.qps)


@dataclass(frozen=True)
class PairedResult:
    baseline: ArmResult
    moduli: ArmResult
    paired_gain_percent: Sequence[float]

    @property
    def paired_median_gain_percent(self) -> float:
        return statistics.median(self.paired_gain_percent)

    @property
    def paired_mean_gain_percent(self) -> float:
        return statistics.mean(self.paired_gain_percent)


def measure_arm(
    query_count: int,
    warmup_rounds: int,
    measured_rounds: int,
    execute_round: Callable[[], float],
    recall_at_k: float,
) -> ArmResult:
    """Measure one arm using the caller's already prepared native path.

    ``execute_round`` must return the complete wall time for one query batch.
    Setup, index loading, and one-time routing are not part of that callback.
    ``query_count`` is retained as an explicit argument to make the QPS
    convention visible at the call site, even though the arm stores time in
    seconds.
    """

    if query_count <= 0 or warmup_rounds < 0 or measured_rounds <= 0:
        raise ValueError("invalid measurement dimensions")
    for _ in range(warmup_rounds):
        execute_round()
    wall_seconds = []
    for _ in range(measured_rounds):
        start = perf_counter()
        reported = execute_round()
        elapsed = float(reported)
        if elapsed <= 0:
            elapsed = perf_counter() - start
        wall_seconds.append(elapsed)
    return ArmResult(query_count, wall_seconds, recall_at_k)


def alternate_compare(
    query_count: int,
    pairs: int,
    warmup_rounds: int,
    rounds_per_arm: int,
    run_baseline: Callable[[], float],
    run_moduli: Callable[[], float],
    baseline_recall: float,
    moduli_recall: float,
) -> PairedResult:
    """Run ``baseline -> MODULI`` and ``MODULI -> baseline`` alternately.

    The callbacks should execute one complete native round and return its wall
    time.  The surrounding runner is responsible for changing only the
    compact-batch state required by the selected arm.
    """

    baseline_times: list[float] = []
    moduli_times: list[float] = []
    for pair in range(pairs):
        order = (run_baseline, run_moduli) if pair % 2 == 0 else (run_moduli, run_baseline)
        first, second = order
        for _ in range(warmup_rounds):
            first()
        for _ in range(rounds_per_arm):
            start = perf_counter()
            elapsed = float(first())
            first_time = elapsed if elapsed > 0 else perf_counter() - start
            (baseline_times if pair % 2 == 0 else moduli_times).append(first_time)
        for _ in range(warmup_rounds):
            second()
        for _ in range(rounds_per_arm):
            start = perf_counter()
            elapsed = float(second())
            second_time = elapsed if elapsed > 0 else perf_counter() - start
            (moduli_times if pair % 2 == 0 else baseline_times).append(second_time)

    baseline = ArmResult(query_count, baseline_times, baseline_recall)
    moduli = ArmResult(query_count, moduli_times, moduli_recall)
    paired = [100.0 * (b / m - 1.0) for b, m in zip(baseline.qps, moduli.qps)]
    return PairedResult(baseline, moduli, paired)
