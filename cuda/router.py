"""CUDA-side logical hierarchy: route, compact, search, and assemble.

The router does not assume that GPU 0 is globally primary.  For each query,
the partition with the smaller coarse distance receives the primary role.  A
locality margin decides whether the secondary partition is needed.  The
result is represented as two dense batches, one per resident index:

``B0 = {primary-0 queries} union {fallback queries}``

``B1 = {primary-1 queries} union {fallback queries}``

Native CAGRA is called at most once per nonempty batch.  Queries that used one
partition take a direct result path.  Only fallback queries enter the two-way
candidate merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .engine import CudaShard, dispatch_pair


Array = np.ndarray


@dataclass(frozen=True)
class RoutePlan:
    """Query-level participation and compact-batch positions."""

    # 0: GPU 0 only, 1: GPU 1 only, 2: both GPUs.
    classes: Array
    batch0_query_ids: Array
    batch1_query_ids: Array
    pos0: Array
    pos1: Array
    margin: Array

    @property
    def dual(self) -> Array:
        return self.classes == 2

    @property
    def secondary_avoidance(self) -> float:
        return (float(np.count_nonzero(self.classes != 2)) / float(self.classes.size)
                if self.classes.size else 0.0)


def _row_distances(queries: Array, centroid: Array) -> Array:
    if queries.ndim != 2 or centroid.ndim != 1:
        raise ValueError("queries must be [n,d] and centroid must be [d]")
    if queries.shape[1] != centroid.shape[0]:
        raise ValueError("centroid dimension does not match queries")
    delta = queries.astype(np.float32, copy=False) - centroid.astype(np.float32, copy=False)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def classify_queries(
    queries: Array,
    centroid0: Array,
    centroid1: Array,
    margin_threshold: float,
) -> RoutePlan:
    """Assign primary roles and fallback participation for a query batch.

    The normalized margin is

    ``abs(d0 - d1) / max(d0, d1)``.

    Small margins are uncertain and retain both partitions.  Ties are assigned
    to GPU 0 only for deterministic behavior; the tie rule is not a fixed
    physical primary role because non-tied queries can select either GPU.
    """

    if margin_threshold < 0:
        raise ValueError("margin_threshold must be non-negative")
    d0 = _row_distances(queries, centroid0)
    d1 = _row_distances(queries, centroid1)
    denominator = np.maximum(np.maximum(d0, d1), np.finfo(np.float32).eps)
    margin = np.abs(d0 - d1) / denominator
    classes = np.where(margin < margin_threshold, 2, np.where(d0 <= d1, 0, 1)).astype(np.uint8)

    query_ids = np.arange(queries.shape[0], dtype=np.int64)
    b0 = query_ids[(classes == 0) | (classes == 2)]
    b1 = query_ids[(classes == 1) | (classes == 2)]
    pos0 = np.full(query_ids.size, -1, dtype=np.int64)
    pos1 = np.full(query_ids.size, -1, dtype=np.int64)
    pos0[b0] = np.arange(b0.size, dtype=np.int64)
    pos1[b1] = np.arange(b1.size, dtype=np.int64)
    return RoutePlan(classes, b0, b1, pos0, pos1, margin)


def compact_queries(queries: Array, query_ids: Array) -> Array:
    """Materialize a dense batch while retaining original query positions."""

    if query_ids.ndim != 1:
        raise ValueError("query_ids must be one-dimensional")
    return np.ascontiguousarray(queries[query_ids])


def _translated_candidates(
    local_neighbors: Array,
    local_distances: Array,
    global_ids: Array,
    row: int,
) -> Tuple[Array, Array]:
    ids = global_ids[local_neighbors[row]]
    distances = local_distances[row]
    return ids.astype(np.uint32, copy=False), distances.astype(np.float32, copy=False)


def _topk(ids: Array, distances: Array, k: int) -> Array:
    """Stable deterministic top-k by distance and then global ID."""

    order = np.lexsort((ids, distances))
    return ids[order[:k]]


def assemble_results(
    plan: RoutePlan,
    neighbors0: Array,
    distances0: Array,
    global_ids0: Array,
    neighbors1: Array,
    distances1: Array,
    global_ids1: Array,
    k: int,
) -> Array:
    """Reconstruct global IDs and preserve the native global top-k contract."""

    if k <= 0:
        raise ValueError("k must be positive")
    n = plan.classes.size
    output = np.empty((n, k), dtype=np.uint32)
    if plan.batch0_query_ids.size:
        if neighbors0.shape[0] != plan.batch0_query_ids.size:
            raise ValueError("GPU 0 result rows do not match its compact batch")
    if plan.batch1_query_ids.size:
        if neighbors1.shape[0] != plan.batch1_query_ids.size:
            raise ValueError("GPU 1 result rows do not match its compact batch")

    only0 = np.flatnonzero(plan.classes == 0)
    only1 = np.flatnonzero(plan.classes == 1)
    dual = np.flatnonzero(plan.classes == 2)
    if only0.size:
        output[only0] = global_ids0[neighbors0[plan.pos0[only0], :k]]
    if only1.size:
        output[only1] = global_ids1[neighbors1[plan.pos1[only1], :k]]
    for query_id in dual:
        p0 = int(plan.pos0[query_id])
        p1 = int(plan.pos1[query_id])
        ids0, dist0 = _translated_candidates(neighbors0, distances0, global_ids0, p0)
        ids1, dist1 = _translated_candidates(neighbors1, distances1, global_ids1, p1)
        ids = np.concatenate((ids0, ids1))
        distances = np.concatenate((dist0, dist1))
        # Partition indexes are disjoint by construction.  The stable sort is
        # still explicit so duplicate or malformed maps cannot make ordering
        # depend on host container iteration order.
        output[query_id] = _topk(ids, distances, k)
    return output


def run_selective_round(
    plan: RoutePlan,
    queries: Array,
    shard0: CudaShard,
    shard1: CudaShard,
    global_ids0: Array,
    global_ids1: Array,
    search_params: object,
    k: int,
) -> Tuple[Array, object]:
    """Execute one already-classified MODULI round.

    Routing and compaction are outside the timed native round.
    The caller may perform them once for a fixed batch and reuse the resulting
    plan across paired rounds.
    """

    shard0.set_batch(compact_queries(queries, plan.batch0_query_ids), k)
    shard1.set_batch(compact_queries(queries, plan.batch1_query_ids), k)
    phase_times = dispatch_pair(shard0, shard1, search_params)
    if shard0.h_neighbors is None or shard0.h_distances is None:
        raise RuntimeError("GPU 0 returned no result buffers")
    if shard1.h_neighbors is None or shard1.h_distances is None:
        raise RuntimeError("GPU 1 returned no result buffers")
    results = assemble_results(
        plan,
        shard0.h_neighbors,
        shard0.h_distances,
        global_ids0,
        shard1.h_neighbors,
        shard1.h_distances,
        global_ids1,
        k,
    )
    return results, phase_times


def run_full_fanout_round(
    queries: Array,
    shard0: CudaShard,
    shard1: CudaShard,
    global_ids0: Array,
    global_ids1: Array,
    search_params: object,
    k: int,
) -> Tuple[Array, object]:
    """The control path using the same resident-shard adapter."""

    classes = np.full(queries.shape[0], 2, dtype=np.uint8)
    # Constructing an explicit all-dual plan keeps result assembly shared with
    # MODULI while avoiding a second merge implementation.
    plan = RoutePlan(
        classes=classes,
        batch0_query_ids=np.arange(queries.shape[0], dtype=np.int64),
        batch1_query_ids=np.arange(queries.shape[0], dtype=np.int64),
        pos0=np.arange(queries.shape[0], dtype=np.int64),
        pos1=np.arange(queries.shape[0], dtype=np.int64),
        margin=np.zeros(queries.shape[0], dtype=np.float32),
    )
    shard0.set_batch(queries, k)
    shard1.set_batch(queries, k)
    phase_times = dispatch_pair(shard0, shard1, search_params)
    if shard0.h_neighbors is None or shard0.h_distances is None:
        raise RuntimeError("GPU 0 returned no result buffers")
    if shard1.h_neighbors is None or shard1.h_distances is None:
        raise RuntimeError("GPU 1 returned no result buffers")
    results = assemble_results(
        plan, shard0.h_neighbors, shard0.h_distances, global_ids0,
        shard1.h_neighbors, shard1.h_distances, global_ids1, k,
    )
    return results, phase_times
