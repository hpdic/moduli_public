#pragma once

// HIP/hipVS execution boundary.  The header exposes the contract needed to
// review the native path while keeping data readers and benchmark entrypoints
// out of the public snapshot.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <raft/core/device_resources.hpp>

#include "moduli/policy.hpp"

namespace moduli {

struct ShardConfig {
  int gpu = 0;
  int shard_id = 0;
  std::string index_path;
  std::string local_to_global_path;
  std::size_t global_start = 0;
};

struct SearchConfig {
  std::size_t k = 10;
  std::size_t local_k = 10;
  std::size_t itopk = 128;
  std::size_t search_width = 1;
  std::size_t repetitions = 1;
  // Use HIP events, stream-ordered D2H copies, and pinned landing buffers.
  // The default remains the conservative synchronized native path.
  bool overlap_copies = false;
};

struct CompactBatch {
  std::vector<uint32_t> original_query_ids;
  std::vector<float> vectors;
  std::size_t dimension = 0;
};

struct NativeBatchResult {
  std::vector<uint32_t> local_ids;
  std::vector<float> distances;
  std::size_t query_count = 0;
  std::size_t local_k = 0;
};

struct RuntimeRound {
  std::vector<uint32_t> global_ids;
  std::vector<float> distances;
  std::vector<int32_t> source_shard;
  double routing_seconds = 0.0;
  double compaction_seconds = 0.0;
  double native_seconds = 0.0;
  double merge_seconds = 0.0;
};

class HipShard;

// Constructing a runtime requires already-built resident indexes.  The
// constructor loads them once and retains ownership on the selected GPU.
class HipRuntime {
 public:
  HipRuntime(std::vector<ShardConfig> shards, SearchConfig search);
  ~HipRuntime();

  HipRuntime(const HipRuntime&) = delete;
  HipRuntime& operator=(const HipRuntime&) = delete;

  RuntimeRound run_selective(const std::vector<float>& queries,
                             std::size_t query_count,
                             std::size_t dimension,
                             const std::vector<float>& centroid0,
                             const std::vector<float>& centroid1,
                             float margin_threshold,
                             SecondaryEffortPolicy effort = {},
                             LoadBias load_bias = {},
                             bool enable_run_ahead = false);

  RuntimeRound run_full_fanout(const std::vector<float>& queries,
                               std::size_t query_count,
                               std::size_t dimension);

 private:
  std::vector<ShardConfig> shards_;
  SearchConfig search_;
  std::vector<std::unique_ptr<HipShard>> resident_;
};

}  // namespace moduli
