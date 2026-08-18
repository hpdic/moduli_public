#include "moduli/policy.hpp"

#include <numeric>

namespace moduli {
namespace {

float distance_to_centroid(const std::vector<float>& queries,
                           std::size_t row, std::size_t dimension,
                           const std::vector<float>& centroid) {
  if (centroid.size() != dimension) {
    throw std::invalid_argument("centroid dimension does not match query dimension");
  }
  double sum = 0.0;
  const auto offset = row * dimension;
  for (std::size_t j = 0; j < dimension; ++j) {
    const double delta = static_cast<double>(queries[offset + j]) - centroid[j];
    sum += delta * delta;
  }
  return static_cast<float>(std::sqrt(sum));
}

}  // namespace

QueryRoute classify_queries(const std::vector<float>& queries,
                            std::size_t query_count,
                            std::size_t dimension,
                            const std::vector<float>& centroid0,
                            const std::vector<float>& centroid1,
                            float margin_threshold) {
  if (dimension == 0 || centroid0.size() != dimension ||
      centroid1.size() != dimension) {
    throw std::invalid_argument("invalid query or centroid dimension");
  }
  if (queries.size() != query_count * dimension) {
    throw std::invalid_argument("query buffer size does not match shape");
  }
  if (margin_threshold < 0.0F) {
    throw std::invalid_argument("margin threshold must be non-negative");
  }

  QueryRoute route;
  route.class_of.resize(query_count);
  route.primary_gpu.resize(query_count);
  route.margin.resize(query_count);
  route.position0.assign(query_count, -1);
  route.position1.assign(query_count, -1);

  for (std::size_t q = 0; q < query_count; ++q) {
    const auto d0 = distance_to_centroid(queries, q, dimension, centroid0);
    const auto d1 = distance_to_centroid(queries, q, dimension, centroid1);
    const auto denominator = std::max(std::max(d0, d1),
                                      std::numeric_limits<float>::epsilon());
    const auto margin = std::abs(d0 - d1) / denominator;
    const auto nearest = d0 <= d1 ? uint8_t{0} : uint8_t{1};
    route.primary_gpu[q] = nearest;
    route.margin[q] = margin;
    route.class_of[q] = margin < margin_threshold
                            ? Participation::dual
                            : (nearest == 0 ? Participation::gpu0_only
                                             : Participation::gpu1_only);
    if (route.class_of[q] != Participation::gpu1_only) {
      route.position0[q] = static_cast<int32_t>(route.batch0_query_ids.size());
      route.batch0_query_ids.push_back(static_cast<uint32_t>(q));
    }
    if (route.class_of[q] != Participation::gpu0_only) {
      route.position1[q] = static_cast<int32_t>(route.batch1_query_ids.size());
      route.batch1_query_ids.push_back(static_cast<uint32_t>(q));
    }
  }
  return route;
}

SecondaryEffortPolicy::SecondaryEffortPolicy(
    bool enabled, uint32_t uniform_cap, std::vector<MarginTier> tiers)
    : enabled_(enabled), uniform_cap_(uniform_cap), tiers_(std::move(tiers)) {
  std::sort(tiers_.begin(), tiers_.end(),
            [](const MarginTier& a, const MarginTier& b) {
              return a.upper_bound < b.upper_bound;
            });
}

uint32_t SecondaryEffortPolicy::choose(float margin,
                                       uint32_t full_iterations) const {
  if (!enabled_) return full_iterations;
  if (!tiers_.empty()) {
    for (const auto& tier : tiers_) {
      if (margin < tier.upper_bound) {
        return tier.secondary_max_iterations == 0
                   ? full_iterations
                   : tier.secondary_max_iterations;
      }
    }
    return full_iterations;
  }
  return uniform_cap_ == 0 ? full_iterations : uniform_cap_;
}

void apply_load_bias(QueryRoute& route, const LoadBias& bias) {
  for (std::size_t q = 0; q < route.class_of.size(); ++q) {
    if (route.class_of[q] != Participation::dual) continue;
    if (bias.should_flip(route.margin[q], route.primary_gpu[q])) {
      route.primary_gpu[q] = static_cast<uint8_t>(bias.primary_gpu);
    }
  }
}

RunAheadLimiter::RunAheadLimiter(int max_slack)
    : max_slack_(std::max(0, max_slack)) {}

bool RunAheadLimiter::admit(int mine, int peer) const {
  if (mine < 0 || peer < 0) return false;
  return mine <= peer + max_slack_;
}

}  // namespace moduli
