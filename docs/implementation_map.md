# Implementation map

The review tree is an extraction of the core runtime path rather
than a copy of either experiment directory.

| Review file | Implementation role | Source provenance |
| --- | --- | --- |
| `cuda/engine.py` | one resident cuVS CAGRA shard, reusable device/host buffers, asynchronous launch/synchronization | `scripts/cuda/moduli_v100_engine.py` |
| `cuda/router.py` | locality margin, primary/secondary classes, compact batches, direct scatter, dual merge | `scripts/cuda/cagra_c1_2gpu_v2.py` and `scripts/cuda/paired_eval_c1_v100_inprocess.py` |
| `cuda/paired_protocol.py` | same-session alternating baseline and MODULI measurement contract | `scripts/cuda/paired_eval_c1_v100_inprocess.py` |
| `hip/include/moduli/policy.hpp` | backend-independent participation and optional effort policies | `moduli_smoke/deep_moduli_c4_pipeline.cpp` |
| `hip/include/moduli/merge.hpp` | global-ID reconstruction and deterministic top-$k$ merge | `moduli_smoke/deep_moduli_batched_locality.cpp` |
| `hip/include/moduli/runtime.hpp` | HIP runtime contract and per-device ownership model | `moduli_smoke/deep_moduli_batched_locality.cpp` |
| `hip/src/runtime.cpp` | HIP/hipVS resident-shard execution and compact-batch dispatch | `moduli_smoke/deep_moduli_batched_locality.cpp` |
| `hip/src/policy.cpp` | C2/C3/C4 policy application and bounded progress control | `moduli_smoke/deep_moduli_c4_pipeline.cpp` |

The source commits used as implementation references are:

- MODULI CUDA: `5bb3185`, `af85905`;
- MODULI HIP: `3da9505`;
- upstream-compatible base: `c0804a5f46fd603794f4ce6205956a35ca46890e`.

The measured deployment configurations were two GPUs, two resident shards,
graph degree 48, sign pruning disabled, CUDA 12.6 on V100 `sm_70`, and the
corresponding HIP/ROCm path on MI100 `gfx908`.  Those deployment facts are
provenance, not build instructions for this review snapshot.
