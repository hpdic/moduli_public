#include "moduli/runtime.hpp"

#include "moduli/merge.hpp"

#include <hip/hip_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <rmm/device_uvector.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace moduli {
namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

void check_hip(hipError_t status, const char* expression) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             hipGetErrorString(status));
  }
}

#define MODULI_HIP_CHECK(expression) \
  ::moduli::check_hip((expression), #expression)

std::vector<uint32_t> read_u32_map(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open global-ID map: " + path);
  input.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::size_t>(input.tellg());
  if (bytes % sizeof(uint32_t) != 0) {
    throw std::runtime_error("global-ID map is not uint32-aligned: " + path);
  }
  std::vector<uint32_t> ids(bytes / sizeof(uint32_t));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(bytes));
  if (!input) throw std::runtime_error("short global-ID map: " + path);
  return ids;
}

uint32_t native_full_iterations(std::size_t itopk, std::size_t search_width,
                                std::size_t min_iterations,
                                std::size_t graph_degree,
                                std::size_t index_size) {
  // This is the resolved full value used when the native scalar parameter is
  // left at its automatic setting.  C2 supplies this value explicitly for
  // primary queries and only lowers it for the secondary role.
  if (search_width == 0) throw std::invalid_argument("search_width is zero");
  uint32_t iterations = static_cast<uint32_t>(itopk / search_width);
  std::size_t reachable = 1;
  const std::size_t branching = std::max<std::size_t>(2, graph_degree / 2);
  while (reachable < index_size) {
    if (reachable > std::numeric_limits<std::size_t>::max() / branching) break;
    reachable *= branching;
    ++iterations;
  }
  return std::max(iterations, static_cast<uint32_t>(min_iterations));
}

struct SearchOutput {
  NativeBatchResult result;
  double elapsed_seconds = 0.0;
};

}  // namespace

class HipShard {
 public:
  explicit HipShard(ShardConfig config) : config_(std::move(config)) {
    MODULI_HIP_CHECK(hipSetDevice(config_.gpu));
    handle_ = std::make_unique<raft::device_resources>();

    indices_ = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(
        *handle_);
    cuvs::neighbors::cagra::deserialize(*handle_, config_.index_path,
                                        indices_.get());
    MODULI_HIP_CHECK(hipDeviceSynchronize());
    if (!config_.local_to_global_path.empty()) {
      local_to_global_ = read_u32_map(config_.local_to_global_path);
    }
  }

  ~HipShard() {
    if (config_.gpu >= 0 && hipSetDevice(config_.gpu) == hipSuccess) {
      if (pinned_neighbors_) hipHostFree(pinned_neighbors_);
      if (pinned_distances_) hipHostFree(pinned_distances_);
    }
  }

  const ShardConfig& config() const { return config_; }
  const std::vector<uint32_t>& local_to_global() const {
    return local_to_global_;
  }

  std::size_t graph_degree() const { return indices_->graph_degree(); }
  std::size_t index_size() const { return indices_->size(); }

  void set_batch(const CompactBatch& batch, std::size_t local_k) {
    MODULI_HIP_CHECK(hipSetDevice(config_.gpu));
    if (batch.dimension == 0 ||
        batch.vectors.size() != batch.original_query_ids.size() * batch.dimension) {
      throw std::invalid_argument("invalid compact HIP batch");
    }
    batch_ = batch;
    local_k_ = local_k;
    if (local_k_ == 0) throw std::invalid_argument("local_k is zero");

    const auto n = static_cast<int64_t>(batch_.original_query_ids.size());
    const auto d = static_cast<int64_t>(batch_.dimension);
    d_queries_ = std::make_unique<raft::device_matrix<float, int64_t>>(
        raft::make_device_matrix<float, int64_t>(*handle_, n, d));
    d_neighbors_ = std::make_unique<raft::device_matrix<uint32_t, int64_t>>(
        raft::make_device_matrix<uint32_t, int64_t>(*handle_, n,
                                                    static_cast<int64_t>(local_k_)));
    d_distances_ = std::make_unique<raft::device_matrix<float, int64_t>>(
        raft::make_device_matrix<float, int64_t>(*handle_, n,
                                                 static_cast<int64_t>(local_k_)));
    const auto required_elements = batch_.original_query_ids.size() * local_k_;
    if (required_elements > pinned_capacity_) {
      if (pinned_neighbors_) {
        MODULI_HIP_CHECK(hipHostFree(pinned_neighbors_));
        pinned_neighbors_ = nullptr;
      }
      if (pinned_distances_) {
        MODULI_HIP_CHECK(hipHostFree(pinned_distances_));
        pinned_distances_ = nullptr;
      }
      MODULI_HIP_CHECK(hipHostMalloc(
          reinterpret_cast<void**>(&pinned_neighbors_),
          required_elements * sizeof(uint32_t)));
      MODULI_HIP_CHECK(hipHostMalloc(
          reinterpret_cast<void**>(&pinned_distances_),
          required_elements * sizeof(float)));
      pinned_capacity_ = required_elements;
    }
    if (!batch_.vectors.empty()) {
      MODULI_HIP_CHECK(hipMemcpy(d_queries_->data_handle(), batch_.vectors.data(),
                                 batch_.vectors.size() * sizeof(float),
                                 hipMemcpyHostToDevice));
    }
  }

  SearchOutput search(const SearchConfig& config,
                      const std::vector<uint32_t>& max_iterations) {
    MODULI_HIP_CHECK(hipSetDevice(config_.gpu));
    if (!d_queries_ || !d_neighbors_ || !d_distances_) {
      throw std::logic_error("set_batch must precede search");
    }
    const auto n = batch_.original_query_ids.size();
    cuvs::neighbors::cagra::search_params params;
    params.itopk_size = config.itopk;
    params.search_width = config.search_width;

    std::unique_ptr<rmm::device_uvector<uint32_t>> d_max_iterations;
    if (!max_iterations.empty()) {
      if (max_iterations.size() != n) {
        throw std::invalid_argument("per-query effort length mismatch");
      }
      d_max_iterations = std::make_unique<rmm::device_uvector<uint32_t>>(
          n, handle_->get_stream());
      MODULI_HIP_CHECK(hipMemcpy(d_max_iterations->data(), max_iterations.data(),
                                 n * sizeof(uint32_t), hipMemcpyHostToDevice));
      // This field is the small MODULI extension in the hipVS CAGRA search
      // parameters.  A null pointer preserves the unmodified native path.
      params.moduli_max_iterations_per_query = d_max_iterations->data();
    }

    const auto start = Clock::now();
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
        d_queries_->data_handle(), static_cast<int64_t>(n),
        static_cast<int64_t>(batch_.dimension));
    auto neighbor_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        d_neighbors_->data_handle(), static_cast<int64_t>(n),
        static_cast<int64_t>(local_k_));
    auto distance_view = raft::make_device_matrix_view<float, int64_t>(
        d_distances_->data_handle(), static_cast<int64_t>(n),
        static_cast<int64_t>(local_k_));
    hipEvent_t event_start = nullptr;
    hipEvent_t event_end = nullptr;
    if (config.overlap_copies && n != 0) {
      // The events and copies use the same RAFT stream.  A stream-ordered D2H
      // path avoids a device-wide synchronization in the host thread while
      // the other physical GPU continues its own native search.
      MODULI_HIP_CHECK(hipEventCreate(&event_start));
      MODULI_HIP_CHECK(hipEventCreate(&event_end));
      MODULI_HIP_CHECK(hipEventRecord(event_start, handle_->get_stream()));
    }
    cuvs::neighbors::cagra::search(*handle_, params, *indices_, query_view,
                                   neighbor_view, distance_view);
    if (config.overlap_copies && n != 0) {
      MODULI_HIP_CHECK(hipEventRecord(event_end, handle_->get_stream()));
      MODULI_HIP_CHECK(hipMemcpyAsync(
          pinned_neighbors_, d_neighbors_->data_handle(),
          n * local_k_ * sizeof(uint32_t), hipMemcpyDeviceToHost,
          handle_->get_stream()));
      MODULI_HIP_CHECK(hipMemcpyAsync(
          pinned_distances_, d_distances_->data_handle(),
          n * local_k_ * sizeof(float), hipMemcpyDeviceToHost,
          handle_->get_stream()));
      MODULI_HIP_CHECK(hipStreamSynchronize(handle_->get_stream()));
    } else {
      MODULI_HIP_CHECK(hipDeviceSynchronize());
    }
    MODULI_HIP_CHECK(hipGetLastError());

    SearchOutput output;
    output.elapsed_seconds = seconds(start, Clock::now());
    if (event_start) {
      float milliseconds = 0.0F;
      MODULI_HIP_CHECK(hipEventElapsedTime(&milliseconds, event_start, event_end));
      output.elapsed_seconds = static_cast<double>(milliseconds) / 1000.0;
    }
    output.result.query_count = n;
    output.result.local_k = local_k_;
    output.result.local_ids.resize(n * local_k_);
    output.result.distances.resize(n * local_k_);
    if (n != 0) {
      if (config.overlap_copies) {
        std::copy_n(pinned_neighbors_, output.result.local_ids.size(),
                    output.result.local_ids.data());
        std::copy_n(pinned_distances_, output.result.distances.size(),
                    output.result.distances.data());
      } else {
        MODULI_HIP_CHECK(hipMemcpy(output.result.local_ids.data(),
                                   d_neighbors_->data_handle(),
                                   output.result.local_ids.size() * sizeof(uint32_t),
                                   hipMemcpyDeviceToHost));
        MODULI_HIP_CHECK(hipMemcpy(output.result.distances.data(),
                                   d_distances_->data_handle(),
                                   output.result.distances.size() * sizeof(float),
                                   hipMemcpyDeviceToHost));
      }
    }
    if (event_start) {
      MODULI_HIP_CHECK(hipEventDestroy(event_start));
      MODULI_HIP_CHECK(hipEventDestroy(event_end));
    }
    return output;
  }

 private:
  ShardConfig config_;
  std::unique_ptr<raft::device_resources> handle_;
  std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> indices_;
  std::vector<uint32_t> local_to_global_;
  CompactBatch batch_;
  std::size_t local_k_ = 0;
  std::unique_ptr<raft::device_matrix<float, int64_t>> d_queries_;
  std::unique_ptr<raft::device_matrix<uint32_t, int64_t>> d_neighbors_;
  std::unique_ptr<raft::device_matrix<float, int64_t>> d_distances_;
  uint32_t* pinned_neighbors_ = nullptr;
  float* pinned_distances_ = nullptr;
  std::size_t pinned_capacity_ = 0;
};

namespace {

CompactBatch make_batch(const std::vector<float>& queries,
                        const std::vector<uint32_t>& query_ids,
                        std::size_t dimension) {
  CompactBatch batch;
  batch.original_query_ids = query_ids;
  batch.dimension = dimension;
  batch.vectors.resize(query_ids.size() * dimension);
  for (std::size_t i = 0; i < query_ids.size(); ++i) {
    const auto source = static_cast<std::size_t>(query_ids[i]) * dimension;
    std::copy_n(queries.data() + source, dimension,
                batch.vectors.data() + i * dimension);
  }
  return batch;
}

std::vector<uint32_t> full_effort(const HipShard& shard,
                                  const SearchConfig& config,
                                  std::size_t count) {
  return std::vector<uint32_t>(
      count, native_full_iterations(config.itopk, config.search_width, 1,
                                    shard.graph_degree(), shard.index_size()));
}

std::vector<uint32_t> role_effort(const HipShard& shard,
                                  const SearchConfig& config,
                                  const CompactBatch& batch,
                                  const QueryRoute& route,
                                  const SecondaryEffortPolicy& policy) {
  // An empty vector tells HipShard to leave the MODULI search-params
  // pointer null, which is the exact native full-effort path.  Only an
  // enabled C2 policy materializes a per-query array.
  if (!policy.enabled()) return {};
  auto values = full_effort(shard, config, batch.original_query_ids.size());
  for (std::size_t i = 0; i < batch.original_query_ids.size(); ++i) {
    const auto q = batch.original_query_ids[i];
    if (route.class_of[q] != Participation::dual) continue;
    if (route.primary_gpu[q] == static_cast<uint8_t>(shard.config().shard_id)) continue;
    values[i] = policy.choose(route.margin[q], values[i]);
  }
  return values;
}

RuntimeRound assemble_round(const QueryRoute& route,
                            const SearchOutput& output0,
                            const HipShard& shard0,
                            const SearchOutput& output1,
                            const HipShard& shard1,
                            std::size_t query_count,
                            std::size_t k,
                            double routing_seconds,
                            double compaction_seconds,
                            double native_seconds) {
  std::vector<CandidateBlock> blocks;
  std::vector<std::size_t> starts(2);
  starts[0] = shard0.config().global_start;
  starts[1] = shard1.config().global_start;
  RuntimeRound result;
  result.global_ids.resize(query_count * k);
  result.distances.resize(query_count * k);
  result.source_shard.resize(query_count * k);
  const auto merge_start = Clock::now();
  for (std::size_t q = 0; q < query_count; ++q) {
    blocks.clear();
    if (route.position0[q] >= 0) {
      blocks.push_back({&output0.result.local_ids, &output0.result.distances,
                        &shard0.local_to_global(), output0.result.query_count,
                        output0.result.local_k,
                        static_cast<std::size_t>(route.position0[q]),
                        shard0.config().shard_id});
    }
    if (route.position1[q] >= 0) {
      blocks.push_back({&output1.result.local_ids, &output1.result.distances,
                        &shard1.local_to_global(), output1.result.query_count,
                        output1.result.local_k,
                        static_cast<std::size_t>(route.position1[q]),
                        shard1.config().shard_id});
    }
    const auto top = merge_query(blocks, k, starts);
    std::copy(top.ids.begin(), top.ids.end(), result.global_ids.begin() + q * k);
    std::copy(top.distances.begin(), top.distances.end(),
              result.distances.begin() + q * k);
    std::copy(top.source_shard.begin(), top.source_shard.end(),
              result.source_shard.begin() + q * k);
  }
  result.routing_seconds = routing_seconds;
  result.compaction_seconds = compaction_seconds;
  result.native_seconds = native_seconds;
  result.merge_seconds = seconds(merge_start, Clock::now());
  return result;
}

}  // namespace

HipRuntime::HipRuntime(std::vector<ShardConfig> shards, SearchConfig search)
    : shards_(std::move(shards)), search_(search) {
  if (shards_.size() != 2) {
    throw std::invalid_argument("the review runtime expects two resident shards");
  }
  if (shards_[0].shard_id != 0 || shards_[1].shard_id != 1 ||
      shards_[0].gpu == shards_[1].gpu) {
    throw std::invalid_argument("resident shards must be logical 0/1 on distinct GPUs");
  }
  if (search_.k == 0 || search_.local_k < search_.k) {
    throw std::invalid_argument("invalid top-k configuration");
  }
  resident_.reserve(shards_.size());
  for (const auto& shard : shards_) {
    resident_.push_back(std::make_unique<HipShard>(shard));
  }
}

HipRuntime::~HipRuntime() = default;

RuntimeRound HipRuntime::run_selective(
    const std::vector<float>& queries, std::size_t query_count,
    std::size_t dimension, const std::vector<float>& centroid0,
    const std::vector<float>& centroid1, float margin_threshold,
    SecondaryEffortPolicy effort, LoadBias load_bias, bool enable_run_ahead) {
  const auto route_start = Clock::now();
  auto route = classify_queries(queries, query_count, dimension, centroid0,
                                centroid1, margin_threshold);
  apply_load_bias(route, load_bias);
  const auto route_end = Clock::now();

  const auto compact_start = Clock::now();
  const auto batch0 = make_batch(queries, route.batch0_query_ids, dimension);
  const auto batch1 = make_batch(queries, route.batch1_query_ids, dimension);
  const auto compact_end = Clock::now();

  resident_[0]->set_batch(batch0, search_.local_k);
  resident_[1]->set_batch(batch1, search_.local_k);
  const auto effort0 = role_effort(*resident_[0], search_, batch0, route, effort);
  const auto effort1 = role_effort(*resident_[1], search_, batch1, route, effort);

  SearchOutput output0, output1;
  const auto native_start = Clock::now();
  const RunAheadLimiter limiter(enable_run_ahead ? 1 : 0);
  std::atomic<int> progress0{0};
  std::atomic<int> progress1{0};
  const auto repetitions = std::max<std::size_t>(1, search_.repetitions);
  auto worker = [&](HipShard& shard, const std::vector<uint32_t>& effort_values,
                    std::atomic<int>& progress, std::atomic<int>& peer_progress,
                    SearchOutput& final_output) {
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
      while (!limiter.admit(static_cast<int>(repetition),
                            peer_progress.load(std::memory_order_acquire))) {
        std::this_thread::yield();
      }
      final_output = shard.search(search_, effort_values);
      progress.store(static_cast<int>(repetition + 1), std::memory_order_release);
    }
  };
  std::thread worker0([&] {
    worker(*resident_[0], effort0, progress0, progress1, output0);
  });
  std::thread worker1([&] {
    worker(*resident_[1], effort1, progress1, progress0, output1);
  });
  worker0.join();
  worker1.join();
  const auto native_end = Clock::now();

  return assemble_round(
      route, output0, *resident_[0], output1, *resident_[1], query_count,
      search_.k, seconds(route_start, route_end),
      seconds(compact_start, compact_end), seconds(native_start, native_end));
}

RuntimeRound HipRuntime::run_full_fanout(const std::vector<float>& queries,
                                         std::size_t query_count,
                                         std::size_t dimension) {
  QueryRoute route;
  route.class_of.assign(query_count, Participation::dual);
  route.primary_gpu.assign(query_count, 0);
  route.margin.assign(query_count, 0.0F);
  route.position0.resize(query_count);
  route.position1.resize(query_count);
  for (std::size_t q = 0; q < query_count; ++q) {
    route.batch0_query_ids.push_back(static_cast<uint32_t>(q));
    route.batch1_query_ids.push_back(static_cast<uint32_t>(q));
    route.position0[q] = static_cast<int32_t>(q);
    route.position1[q] = static_cast<int32_t>(q);
  }
  const auto batch0 = make_batch(queries, route.batch0_query_ids, dimension);
  const auto batch1 = make_batch(queries, route.batch1_query_ids, dimension);
  resident_[0]->set_batch(batch0, search_.local_k);
  resident_[1]->set_batch(batch1, search_.local_k);
  SearchOutput output0, output1;
  const auto start = Clock::now();
  std::thread worker0([&] { output0 = resident_[0]->search(search_, {}); });
  std::thread worker1([&] { output1 = resident_[1]->search(search_, {}); });
  worker0.join();
  worker1.join();
  return assemble_round(route, output0, *resident_[0], output1, *resident_[1],
                        query_count, search_.k, 0.0, 0.0,
                        seconds(start, Clock::now()));
}

}  // namespace moduli
