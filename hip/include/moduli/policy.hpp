#pragma once

// Query-level policy for the HIP implementation.
//
// The policy layer has no HIP or cuVS dependency.  That separation is what
// lets the same logical hierarchy be applied to different native search
// backends without moving resident indexes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moduli {

enum class Participation : uint8_t {
  gpu0_only = 0,
  gpu1_only = 1,
  dual = 2,
};

struct QueryRoute {
  std::vector<Participation> class_of;
  std::vector<uint8_t> primary_gpu;
  std::vector<float> margin;
  std::vector<uint32_t> batch0_query_ids;
  std::vector<uint32_t> batch1_query_ids;
  std::vector<int32_t> position0;
  std::vector<int32_t> position1;
};

// The score is a coarse locality score.  It selects
// participation; it does not replace native graph traversal or promise that
// the selected partition contains every true neighbor.
QueryRoute classify_queries(const std::vector<float>& queries,
                            std::size_t query_count,
                            std::size_t dimension,
                            const std::vector<float>& centroid0,
                            const std::vector<float>& centroid1,
                            float margin_threshold);

struct MarginTier {
  float upper_bound = 1.0F;
  uint32_t secondary_max_iterations = 0;
};

// C2 policy: a value of zero means use the native full-effort value.  Tiers
// are ordered by upper_bound and are consulted only for dual queries on the
// secondary role.  Single-partition and primary-role queries remain full
// effort.
class SecondaryEffortPolicy {
 public:
  SecondaryEffortPolicy() = default;
  SecondaryEffortPolicy(bool enabled, uint32_t uniform_cap,
                        std::vector<MarginTier> tiers);

  uint32_t choose(float margin, uint32_t full_iterations) const;
  bool enabled() const { return enabled_; }

 private:
  bool enabled_ = false;
  uint32_t uniform_cap_ = 0;
  std::vector<MarginTier> tiers_;
};

// C3 can change the logical primary role only inside a measured, narrow
// near-tie band.  It never changes which GPUs participate.
struct LoadBias {
  int primary_gpu = -1;
  float margin_epsilon = 0.0F;

  bool should_flip(float margin, int nearest_gpu) const {
    return primary_gpu >= 0 && margin < margin_epsilon &&
           nearest_gpu != primary_gpu;
  }
};

void apply_load_bias(QueryRoute& route, const LoadBias& bias);

// C4 is a host admission rule, not an unbounded producer queue.  A worker may
// submit the next repetition only while its progress is within max_slack of
// the slowest peer.  max_slack == 0 means lock-step; max_slack == 1 permits
// one outstanding repetition.
class RunAheadLimiter {
 public:
  explicit RunAheadLimiter(int max_slack);

  bool admit(int mine, int peer) const;
  int max_slack() const { return max_slack_; }

 private:
  int max_slack_;
};

}  // namespace moduli
