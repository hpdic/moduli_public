"""The CUDA/cuVS resident-shard execution boundary.

This module contains the part of MODULI that is specific to the CUDA native
search path.  It does not build an index and it does not decide how a query is
routed.  A :class:`CudaShard` owns one resident cuVS CAGRA index, one explicit
cuVS resource object, and reusable device and host buffers for its current
compact batch.  The caller can therefore launch both shards before waiting on
either one.

The separation is important for review: ``CudaShard`` is the native execution
adapter, while ``router.py`` contains the logical memory-hierarchy policy.
Both the full-fan-out control path and the selective path use this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, MutableMapping, Optional, Tuple

import cupy as cp
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import cagra


Array = np.ndarray


@dataclass
class PhaseTimes:
    """Host-visible portions of one two-GPU dispatch."""

    launch_s: float
    synchronize_s: float
    fetch_s: float


@dataclass
class CudaShard:
    """A resident CAGRA partition and its reusable per-batch state.

    ``index_path`` is supplied by the caller because index construction and
    storage layout are outside the review release.  The object never moves an
    index between GPUs.  The ``gpu`` argument is the physical device that owns
    the partition for the lifetime of the object.
    """

    gpu: int
    index_path: str
    index: Any = field(init=False)
    resources: Resources = field(init=False)
    query_count: int = field(default=0, init=False)
    k: int = field(default=0, init=False)
    d_queries: Optional[Any] = field(default=None, init=False)
    d_neighbors: Optional[Any] = field(default=None, init=False)
    d_distances: Optional[Any] = field(default=None, init=False)
    h_neighbors: Optional[Array] = field(default=None, init=False)
    h_distances: Optional[Array] = field(default=None, init=False)

    def __post_init__(self) -> None:
        with cp.cuda.Device(self.gpu):
            self.index = cagra.load(self.index_path)
            # An explicit resource object makes launch and synchronization
            # ownership visible.  The object is never shared across GPUs.
            self.resources = Resources()

    def set_batch(self, host_queries: Array, k: int) -> None:
        """Upload one dense compact batch and allocate its output buffers.

        The batch is fixed while a measurement arm runs.  Reusing all buffers
        across rounds removes allocator churn from the search measurement and
        keeps the baseline and MODULI paths on the same native adapter.
        """

        if host_queries.ndim != 2:
            raise ValueError("host_queries must have shape [queries, dimension]")
        if k <= 0:
            raise ValueError("k must be positive")

        self.query_count = int(host_queries.shape[0])
        self.k = int(k)
        if self.query_count == 0:
            self.d_queries = None
            self.d_neighbors = None
            self.d_distances = None
            self.h_neighbors = None
            self.h_distances = None
            return

        with cp.cuda.Device(self.gpu):
            self.d_queries = cp.asarray(host_queries)
            self.d_neighbors = cp.empty((self.query_count, self.k), dtype=cp.uint32)
            self.d_distances = cp.empty((self.query_count, self.k), dtype=cp.float32)

        self.h_neighbors = np.empty((self.query_count, self.k), dtype=np.uint32)
        self.h_distances = np.empty((self.query_count, self.k), dtype=np.float32)

    def launch(self, search_params: Any) -> None:
        """Issue the native search without synchronizing the host."""

        if self.query_count == 0:
            return
        with cp.cuda.Device(self.gpu):
            cagra.search(
                search_params,
                self.index,
                self.d_queries,
                self.k,
                neighbors=self.d_neighbors,
                distances=self.d_distances,
                resources=self.resources,
            )

    def synchronize(self) -> None:
        """Wait for this shard's explicit cuVS resource."""

        if self.query_count:
            self.resources.sync()

    def fetch(self) -> Tuple[Optional[Array], Optional[Array]]:
        """Copy results into stable host buffers after synchronization."""

        if self.query_count == 0:
            return None, None
        if self.h_neighbors is None or self.h_distances is None:
            raise RuntimeError("set_batch must be called before fetch")
        with cp.cuda.Device(self.gpu):
            self.d_neighbors.get(out=self.h_neighbors)
            self.d_distances.get(out=self.h_distances)
        return self.h_neighbors, self.h_distances


def dispatch_pair(
    shard0: CudaShard,
    shard1: CudaShard,
    search_params: Any,
    profile: Optional[MutableMapping[str, list[float]]] = None,
) -> PhaseTimes:
    """Launch, synchronize, and fetch two compact batches.

    The launch phase issues both native calls from one host
    thread.  cuVS receives separate resources and therefore retains device
    concurrency without making Python thread scheduling part of the critical
    path.  The returned interval includes the result transfers because the
    public comparison contract measures a complete query round.
    """

    t0 = perf_counter()
    shard0.launch(search_params)
    shard1.launch(search_params)
    t1 = perf_counter()

    shard0.synchronize()
    shard1.synchronize()
    t2 = perf_counter()

    shard0.fetch()
    shard1.fetch()
    t3 = perf_counter()
    times = PhaseTimes(t1 - t0, t2 - t1, t3 - t2)
    if profile is not None:
        profile.setdefault("launch_s", []).append(times.launch_s)
        profile.setdefault("synchronize_s", []).append(times.synchronize_s)
        profile.setdefault("fetch_s", []).append(times.fetch_s)
    return times


def make_search_params(
    *,
    itopk_size: int,
    search_width: int = 1,
    algo: str = "auto",
) -> Any:
    """Create a cuVS CAGRA parameter object without fixing a dataset path."""

    if itopk_size <= 0 or search_width <= 0:
        raise ValueError("itopk_size and search_width must be positive")
    return cagra.SearchParams(
        itopk_size=itopk_size,
        search_width=search_width,
        algo=algo,
    )
