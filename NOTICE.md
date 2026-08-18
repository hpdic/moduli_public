# Notices and dependency boundary

The HIP implementation is an adaptation of the MODULI runtime work on top of
the HIP/hipVS and RAFT APIs.  The upstream hipVS tree is Apache License 2.0;
the complete upstream license and notices remain with that dependency and are
not reproduced as a second copy of the upstream project here.

The CUDA implementation calls the public CuPy and cuVS APIs.  Their licenses,
source, and distribution terms remain those of their respective projects.

This review snapshot contains only MODULI-specific orchestration code.  It
does not redistribute cuVS, hipVS, RAFT, CuPy, CUDA, ROCm, native indexes,
datasets, or benchmark artifacts.

The release license for this review snapshot will be finalized before the
repository is made public.  Until then, the files are supplied to paper
reviewers for inspection only.
