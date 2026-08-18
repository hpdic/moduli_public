# Review scope

The public review snapshot follows the paper's runtime boundary.

## Included mechanisms

1. **Selective participation.**  The router scores the two resident
   partitions, chooses a query-dependent primary role, and retains both
   partitions for low-margin queries.
2. **Batch-preserving execution.**  Per-query decisions become one dense
   compact batch per active GPU.  Empty batches skip native work; the native
   engine is not called once per query.
3. **Global result semantics.**  Local neighbor IDs are translated through
   the partition map and merged by distance, global ID, and source shard.
4. **Asymmetric effort.**  The secondary role can receive a lower per-query
   iteration cap.  A null effort pointer preserves the native full-effort
   path.
5. **Load-aware roles and bounded progress.**  The near-tie load-bias rule
   changes only the logical primary role.  The run-ahead limiter bounds the
   distance between worker progress values.

## Not included

- vector, graph, checkpoint, centroid, or ground-truth construction;
- dataset download, partition manifests, and global-ID-map generation;
- private cluster paths, hostnames, credentials, and environment modules;
- complete benchmark drivers, sweep scripts, result archival, and plotting;
- large graph/index/checkpoint artifacts;
- rejected kernel prototypes and platform-specific tuning attempts;
- the release packaging and reproducibility automation planned after
  acceptance.

These omissions let reviewers inspect the algorithmic and systems
implementation without receiving the full internal experiment
repository or a promise of turnkey execution from this snapshot alone.
