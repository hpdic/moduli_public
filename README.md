# MODULI review source

This repository contains the implementation-facing core of **MODULI
(Multi-GPU Orchestration of Data Using Locality-aware Indexing)** for paper
review.  The snapshot is organized as two backend lines:

- [`cuda/`](cuda/) — the CUDA/cuVS execution path;
- [`hip/`](hip/) — the HIP/hipVS execution path.

The released material is limited to the mechanisms needed to
inspect the system design: query-locality classification, primary/secondary
participation, dense batch formation, native search dispatch, result-ID
translation, global top-$k$ assembly, and the optional policy hooks described
in the paper.  Dataset manifests, graph/index construction, large artifacts,
cluster launch scripts, and the complete executable benchmark harness are not
included in this review snapshot.  The snapshot is therefore not advertised
as a turnkey reproduction package and is not expected to run without the
unreleased data and build integration.  A complete executable release will be
considered after paper acceptance.

The source is provided for technical review.  In particular, the two backend
directories show where the common MODULI contract meets the different native
runtime paths; they are not two copies of a hidden full application.

## What is included

The CUDA line exposes a reusable asynchronous cuVS CAGRA shard wrapper, the
two-GPU router and compaction logic, the primary-only fast path, the dual-GPU
merge path, and a paired-comparison protocol.  The HIP line exposes the same
runtime contract as C++ interfaces, including per-device resource ownership,
compacted query batches, CAGRA dispatch, global-ID reconstruction, dense
result merging, and policy decisions for asymmetric effort, load-aware role
assignment, and bounded run-ahead.

The implementation assumes that the native partition indexes and their
partition-to-global-ID maps have already been built.  Building those objects
is outside this repository.

## Provenance

The review snapshot is derived from the MODULI CUDA implementation in the
`moduli` repository and the HIP implementation in the `moduli_hipVS` fork.
The HIP line is based on the Apache-licensed hipVS/RAFT software stack; see
[`NOTICE.md`](NOTICE.md) for the dependency boundary.  This repository does
not copy the full upstream projects.

The implementation provenance used for the paper is recorded in
[`docs/implementation_map.md`](docs/implementation_map.md).

## Review boundary

This is a source-disclosure snapshot for reviewers.  It is not a promise that
the checked-in files alone reproduce every table in the paper.  Reproduction
requires the withheld index artifacts, dataset-specific manifests, build
configuration, and experiment orchestration.  No credentials, cluster paths,
large binary artifacts, or private run logs belong in this repository.
