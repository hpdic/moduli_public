cmake_minimum_required(VERSION 3.24)
project(moduli_review_hip LANGUAGES CXX)

# This file records the intended library boundary.  It is not a complete
# dependency/bootstrap script: the review snapshot omits the
# private ROCm/hipVS build and dataset integration.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(moduli_hip_review STATIC
  src/policy.cpp
  src/runtime.cpp
)
target_include_directories(moduli_hip_review PUBLIC include)

# A site-specific build must provide HIP, RAFT, RMM, and hipVS/cuVS targets.
# The names below match the development tree used for the paper; no download
# or vendoring is performed by this repository.
find_package(hip REQUIRED)
target_link_libraries(moduli_hip_review PUBLIC hip::host raft cuvs)
