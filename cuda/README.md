# CUDA line

The CUDA line uses one cuVS CAGRA index per GPU.  `engine.py` owns the native
resources and reusable buffers.  `router.py` decides which compact batch each
GPU receives and performs the primary-only or dual-candidate result path.
`paired_protocol.py` documents the measurement boundary without embedding
dataset-specific paths or the private benchmark harness.

The implementation uses explicit `cuvs.common.Resources` objects.  A single
host thread launches both asynchronous searches before synchronizing either
resource.  This keeps dispatch ordering visible and avoids making Python
thread scheduling part of the measured multi-GPU path.

The files require CuPy, NumPy, and cuVS for execution.  They do not
not provide index construction, data download, ground-truth generation, or a
paper-result command line.
