# HIP line

The HIP line is a C++ runtime-facing extraction for hipVS/CAGRA.  The runtime
keeps one `raft::device_resources` object and one native execution context per
physical GPU worker.  A worker loads resident indexes, uploads its compacted
query batch once, reuses result buffers, dispatches native CAGRA searches, and
returns local IDs for global assembly.

`policy.hpp` contains the query-level contract.  `runtime.hpp` defines the
backend boundary, `merge.hpp` defines deterministic global-ID assembly, and
`src/runtime.cpp` shows the HIP resource and synchronization path.  The policy
implementation keeps C2/C3/C4 conditional and disabled unless the caller
selects a policy explicitly.

The runtime also contains the C4 copy path: HIP events, stream-ordered
device-to-host transfers, and reusable pinned landing buffers.  The
conservative path uses device synchronization and ordinary host copies, which
keeps the control boundary explicit.

The files require a compatible HIP, RAFT, and hipVS/cuVS development
environment.  Dataset readers, index builders, benchmark launchers, and
large artifacts are not included.
