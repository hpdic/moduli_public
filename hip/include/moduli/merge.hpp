#pragma once

// Host-side global-ID reconstruction and deterministic candidate assembly.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moduli {

struct CandidateBlock {
  // Row-major [compact_query_count, local_k].
  const std::vector<uint32_t>* local_ids = nullptr;
  const std::vector<float>* distances = nullptr;
  const std::vector<uint32_t>* local_to_global = nullptr;
  std::size_t compact_query_count = 0;
  std::size_t local_k = 0;
  std::size_t original_position = 0;
  int shard_id = -1;
};

struct MergedTopK {
  std::vector<uint32_t> ids;
  std::vector<float> distances;
  std::vector<int32_t> source_shard;
};

inline uint32_t translate_id(const CandidateBlock& block, uint32_t local_id,
                             std::size_t global_start) {
  if (block.local_to_global && !block.local_to_global->empty()) {
    if (local_id >= block.local_to_global->size()) {
      throw std::out_of_range("native local ID is outside the ID map");
    }
    return (*block.local_to_global)[local_id];
  }
  const auto global = static_cast<uint64_t>(global_start) + local_id;
  if (global > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("global ID does not fit in uint32_t");
  }
  return static_cast<uint32_t>(global);
}

inline MergedTopK merge_query(const std::vector<CandidateBlock>& blocks,
                              std::size_t k,
                              const std::vector<std::size_t>& global_starts) {
  if (k == 0) throw std::invalid_argument("k must be positive");
  std::vector<std::tuple<float, uint32_t, int32_t>> candidates;
  for (const auto& block : blocks) {
    if (!block.local_ids || !block.distances || block.local_k == 0) continue;
    if (block.original_position >= block.compact_query_count) continue;
    const auto row = block.original_position * block.local_k;
    for (std::size_t j = 0; j < block.local_k; ++j) {
      const auto local = (*block.local_ids)[row + j];
      const auto gid = translate_id(block, local, global_starts.at(block.shard_id));
      candidates.emplace_back((*block.distances)[row + j], gid,
                              static_cast<int32_t>(block.shard_id));
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
              if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
              if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) < std::get<1>(b);
              return std::get<2>(a) < std::get<2>(b);
            });

  MergedTopK output;
  output.ids.reserve(k);
  output.distances.reserve(k);
  output.source_shard.reserve(k);
  std::unordered_set<uint32_t> seen;
  for (const auto& candidate : candidates) {
    if (output.ids.size() == k) break;
    if (!seen.insert(std::get<1>(candidate)).second) continue;
    output.distances.push_back(std::get<0>(candidate));
    output.ids.push_back(std::get<1>(candidate));
    output.source_shard.push_back(std::get<2>(candidate));
  }
  while (output.ids.size() < k) {
    output.ids.push_back(std::numeric_limits<uint32_t>::max());
    output.distances.push_back(std::numeric_limits<float>::infinity());
    output.source_shard.push_back(-1);
  }
  return output;
}

}  // namespace moduli
