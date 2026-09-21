#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// inline void check_cuda(cudaError_t err, const char *expression,
//                        const char *file, int line) {
//   if (err == cudaSuccess) {
//     return;
//   }

//   std::fprintf(stderr,
//                "%s:%d: %s failed\n"
//                "  %s (%d): %s\n",
//                file, line, expression, cudaGetErrorName(err),
//                static_cast<int>(err), cudaGetErrorString(err));

//   std::exit(EXIT_FAILURE);
// }

static void check_cuda(cudaError_t error, const char *expression, int line) {
  if (error != cudaSuccess)
    throw std::runtime_error("CUDA line " + std::to_string(line) + ": " +
                             expression + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

constexpr int kBlock = 128;

struct Config {
  int cells = 16;
  int steps = 10000;
  int warmup = 1000;
  int thermo = 1000;
  int dump = 0;
  std::string output = "out.xyz";
  int capacity = 0;
  int lanes = 4;
  unsigned seed = 42;
  double density = 0.8;
  double temperature = 1.0;
  double dt = 0.002;
  double cutoff = 2.5;
  double skin = 0.4;
  bool check = false;
  bool graphs = true;
  bool rebuild_every_step = false;
};

struct Params {
  int n, pitch, nc, cell_count, capacity, rebuild_every_step;
  float box, inv_box, inv_cell, cutoff2, list2, rebuild2, dt;
  double shift;
};

// struct Particle {
//   float4 position, velocity, force;
// };
// std::vector<Particle> particles;

enum Error : unsigned { kOverflow = 1, kNonfinite = 2, kOverlap = 4 };

struct Status {
  unsigned error = 0;
  int rebuild = 0;
  int max_neighbors = 0;
  unsigned long long builds = 0;
};

struct DeviceState {
  float4 *position, *velocity, *alternate_position, *alternate_velocity;
  float4 *force, *displacement;
  int *cell_id, *cell_counts, *cell_offsets;
  int *neighbors, *neighbor_count;
  Status *status;
};

template <class T> struct DeviceBuffer {
  T *ptr = nullptr;
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  void allocate(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
};

// CUDA_CHECK(cudaMalloc(&position, n * sizeof(float4)));
// CUDA_CHECK(cudaFree(position));

struct Totals {
  double kinetic, potential, virial, px, py, pz, neighbors;
};

struct AddTotals {
  __host__ __device__ Totals operator()(const Totals &a,
                                        const Totals &b) const {
    return {a.kinetic + b.kinetic,
            a.potential + b.potential,
            a.virial + b.virial,
            a.px + b.px,
            a.py + b.py,
            a.pz + b.pz,
            a.neighbors + b.neighbors};
  }
};

__device__ __forceinline__ float wrap(float x, const Params &p) {
  // if (x < 0) x += p.box; if (x >= p.box) x -= p.box;
  x -= p.box * floorf(x * p.inv_box);
  if (x >= p.box)
    x = 0.0f;
  if (x < 0.0f)
    x += p.box;
  return x;
}

__device__ __forceinline__ float image(float dx, const Params &p) {
  return dx - p.box * nearbyintf(dx * p.inv_box);
}

__device__ __forceinline__ int cell_id(float4 r, const Params &p) {
  int x = min(static_cast<int>(r.x * p.inv_cell), p.nc - 1);
  // int(r.x / (p.box / p.nc));
  int y = min(static_cast<int>(r.y * p.inv_cell), p.nc - 1);
  // int(r.y / (p.box / p.nc));
  int z = min(static_cast<int>(r.z * p.inv_cell), p.nc - 1);
  return (z * p.nc + y) * p.nc + x;
}

__device__ __forceinline__ int cell_wrap(int x, int n) {
  if (x < 0)
    x += n;
  if (x >= n)
    x -= n;
  return x;
}

__global__ void count_cells(DeviceState *state, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n)
    return;
  int cell = cell_id(state->position[i], p);
  state->cell_id[i] = cell;
  atomicAdd(state->cell_counts + cell, 1);
}

__global__ void scatter_particles(DeviceState *state, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n)
    return;
  int cell = state->cell_id[i];
  int j = state->cell_offsets[cell] + atomicAdd(state->cell_counts + cell, 1);
  state->alternate_position[j] = state->position[i];
  state->alternate_velocity[j] = state->velocity[i];
  // state->position[j] = state->position[i];
}

__global__ void swap_particles(DeviceState *state) {
  // cudaMemcpyAsync(pos.ptr, alt_pos.ptr, bytes, cudaMemcpyDeviceToDevice,
  // stream); cudaMemcpyAsync(vel.ptr, alt_vel.ptr, bytes,
  // cudaMemcpyDeviceToDevice, stream);
  float4 *tmp = state->position;
  state->position = state->alternate_position;
  state->alternate_position = tmp;
  tmp = state->velocity;
  state->velocity = state->alternate_velocity;
  state->alternate_velocity = tmp;
}

__global__ void build_neighbors(DeviceState *state, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  if (i < p.n) {
    const float4 *position = state->position;
    float4 ri = position[i];
    int cell = cell_id(ri, p);
    int cx = cell % p.nc;
    int cy = (cell / p.nc) % p.nc;
    int cz = cell / (p.nc * p.nc);

    int span = min(3, p.nc);
    for (int z = 0; z < span; z++) {
      int iz = cell_wrap(cz + z - 1, p.nc);
      for (int y = 0; y < span; y++) {
        int iy = cell_wrap(cy + y - 1, p.nc);
        for (int x = 0; x < span; x++) {
          int ix = cell_wrap(cx + x - 1, p.nc);
          int c = (iz * p.nc + iy) * p.nc + ix;
          int end = state->cell_offsets[c + 1];
          for (int j = state->cell_offsets[c]; j < end; j++) {
            float4 rj = __ldg(position + j);
            float dx = image(ri.x - rj.x, p);
            float dy = image(ri.y - rj.y, p);
            float dz = image(ri.z - rj.z, p);
            float r2 = dx * dx + dy * dy + dz * dz;
            // if (j != i && r2 < p.cutoff2)
            // neighbors[size_t(i) * p.capacity + count] = j;
            if (j != i && r2 < p.list2) {
              if (count < p.capacity)
                state->neighbors[static_cast<size_t>(count) * p.pitch + i] = j;
              count++;
            }
          }
        }
      }
    }
    state->neighbor_count[i] = count;
    state->displacement[i] = make_float4(0, 0, 0, 0);
    if (count > p.capacity)
      atomicOr(&state->status->error, kOverflow);
  }
  using Reduce = cub::BlockReduce<int, kBlock>;
  __shared__ typename Reduce::TempStorage scratch;
  int maximum = Reduce(scratch).Reduce(count, cub::Max());
  // if (i < p.n) atomicMax(&state->status->max_neighbors, count);
  if (threadIdx.x == 0)
    atomicMax(&state->status->max_neighbors, maximum);
}

__global__ void finish_build(DeviceState *state) {
  state->status->rebuild = 0;
  state->status->builds++;
}

__global__ void drift(DeviceState *state, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int rebuild = 0;
  if (i < p.n && !state->status->error) {
    float4 r = state->position[i];
    float4 v = state->velocity[i];
    float4 f = state->force[i];
    float4 d = state->displacement[i];
    v.x += 0.5f * p.dt * f.x;
    v.y += 0.5f * p.dt * f.y;
    v.z += 0.5f * p.dt * f.z;
    float dx = p.dt * v.x, dy = p.dt * v.y, dz = p.dt * v.z;
    // d.x = r.x - position_at_last_build[i].x;
    d.x += dx;
    d.y += dy;
    d.z += dz;
    r.x = wrap(r.x + dx, p);
    r.y = wrap(r.y + dy, p);
    r.z = wrap(r.z + dz, p);
    if (!isfinite(r.x + r.y + r.z) || !isfinite(v.x + v.y + v.z) ||
        !isfinite(d.x + d.y + d.z)) {
      atomicOr(&state->status->error, kNonfinite);
    } else {
      state->position[i] = r;
      state->velocity[i] = v;
      state->displacement[i] = d;
      rebuild = p.rebuild_every_step ||
                d.x * d.x + d.y * d.y + d.z * d.z >= p.rebuild2;
    }
  }
  int needed = __syncthreads_or(rebuild);
  // if (rebuild) atomicExch(&state->status->rebuild, 1);
  if (needed && threadIdx.x == 0)
    atomicExch(&state->status->rebuild, 1);
}

__global__ void choose_rebuild(DeviceState *state,
                               cudaGraphConditionalHandle handle) {
  cudaGraphSetConditional(handle,
                          state->status->rebuild && !state->status->error);
}

template <int Lanes>
__global__ __launch_bounds__(kBlock) void force_kick(DeviceState *state,
                                                     Params p, float kick) {
  // int i = blockIdx.x * blockDim.x + threadIdx.x;
  // for (int k = 0; k < state->neighbor_count[i]; k++)
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int i = thread / Lanes;
  int lane = threadIdx.x % Lanes;
  if (i >= p.n || state->status->error)
    return;
  unsigned mask = __activemask();
  const float4 *__restrict__ position = state->position;
  float4 ri = __ldg(position + i);
  float fx = 0, fy = 0, fz = 0;
  int count = min(state->neighbor_count[i], p.capacity);
  for (int k = lane; k < count; k += Lanes) {
    int j = __ldg(state->neighbors + static_cast<size_t>(k) * p.pitch + i);
    float4 rj = __ldg(position + j);
    float dx = image(ri.x - rj.x, p);
    float dy = image(ri.y - rj.y, p);
    float dz = image(ri.z - rj.z, p);
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < p.cutoff2) {
      if (r2 == 0.0f) {
        atomicOr(&state->status->error, kOverlap);
        continue;
      }
      // float r = sqrtf(r2);
      // float derivative = 48.0f * (-powf(r, -13) + .5f * powf(r, -7));
      // float scale = -derivative / r;
      float inv2 = __frcp_rn(r2);
      float inv6 = inv2 * inv2 * inv2;
      float scale = 24.0f * inv2 * inv6 * (2.0f * inv6 - 1.0f);
      // fx += scale * dx;
      // fy += scale * dy;
      // fz += scale * dz;
      fx = fmaf(scale, dx, fx);
      fy = fmaf(scale, dy, fy);
      fz = fmaf(scale, dz, fz);
    }
  }
#pragma unroll
  for (int offset = Lanes / 2; offset > 0; offset /= 2) {
    fx += __shfl_down_sync(mask, fx, offset, Lanes);
    fy += __shfl_down_sync(mask, fy, offset, Lanes);
    fz += __shfl_down_sync(mask, fz, offset, Lanes);
  }
  // atomicAdd(&state->force[i].x, fx);
  // atomicAdd(&state->force[i].y, fy);
  // atomicAdd(&state->force[i].z, fz);
  if (lane == 0) {
    if (!isfinite(fx) || !isfinite(fy) || !isfinite(fz))
      atomicOr(&state->status->error, kNonfinite);
    state->force[i] = make_float4(fx, fy, fz, 0);
    // finish_kick
    float4 v = state->velocity[i];
    v.x += kick * fx;
    v.y += kick * fy;
    v.z += kick * fz;
    state->velocity[i] = v;
  }
}

__global__ void observe(DeviceState *state, Params p, Totals *blocks) {
  // double u = 4.0 * inv6 * (inv6 - 1.0) - p.shift;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  Totals t{};
  if (i < p.n && !state->status->error) {
    float4 ri = state->position[i], v = state->velocity[i];
    t.kinetic =
        0.5 * (double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
    t.px = v.x;
    t.py = v.y;
    t.pz = v.z;
    t.neighbors = state->neighbor_count[i];
    for (int k = 0; k < state->neighbor_count[i]; k++) {
      int j = state->neighbors[static_cast<size_t>(k) * p.pitch + i];
      float4 rj = state->position[j];
      double dx = image(ri.x - rj.x, p);
      double dy = image(ri.y - rj.y, p);
      double dz = image(ri.z - rj.z, p);
      double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < p.cutoff2 && r2 > 0.0) {
        double inv2 = 1.0 / r2;
        double inv6 = inv2 * inv2 * inv2;
        t.potential += 0.5 * (4.0 * inv6 * (inv6 - 1.0) - p.shift);
        t.virial += 12.0 * inv6 * (2.0 * inv6 - 1.0);
      }
    }
  }
  using Reduce = cub::BlockReduce<Totals, kBlock>;
  __shared__ typename Reduce::TempStorage scratch;
  Totals sum = Reduce(scratch).Reduce(t, AddTotals{});
  // atomicAdd(&global_totals->kinetic, t.kinetic);
  if (threadIdx.x == 0)
    blocks[blockIdx.x] = sum;
}

struct InitialState {
  float box;
  std::vector<float4> position, velocity;
};

static InitialState initialize(const Config &c, bool perturb = false) {
  int n = 4 * c.cells * c.cells * c.cells;
  InitialState init;
  init.box = static_cast<float>(std::cbrt(n / c.density));
  init.position.resize(n);
  init.velocity.resize(n);
  constexpr double basis[4][3] = {
      {0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
  // cube™
  // for (int i = 0; i < n; i++)
  //   position[i] = make_float4((i % m) * spacing,
  //                            ((i / m) % m) * spacing,
  //                            (i / (m*m)) * spacing, float(i));
  double spacing = double(init.box) / c.cells;
  std::mt19937 rng(c.seed);
  std::normal_distribution<double> gaussian;
  std::uniform_real_distribution<double> jitter(-0.02, 0.02);
  auto periodic = [&](double x) {
    x -= init.box * std::floor(x / init.box);
    float r = static_cast<float>(x);
    return r >= init.box ? 0.0f : r;
  };
  int i = 0;
  double vx = 0, vy = 0, vz = 0;
  for (int z = 0; z < c.cells; z++)
    for (int y = 0; y < c.cells; y++)
      for (int x = 0; x < c.cells; x++)
        for (const auto &b : basis) {
          double jx = perturb ? jitter(rng) + .137 : 0;
          double jy = perturb ? jitter(rng) - .193 : 0;
          double jz = perturb ? jitter(rng) + .071 : 0;
          init.position[i] =
              make_float4(periodic((x + b[0]) * spacing + jx),
                          periodic((y + b[1]) * spacing + jy),
                          periodic((z + b[2]) * spacing + jz), float(i));
          init.velocity[i] =
              make_float4(gaussian(rng), gaussian(rng), gaussian(rng), 0);
          vx += init.velocity[i].x;
          vy += init.velocity[i].y;
          vz += init.velocity[i].z;
          i++;
        }
  double speed2 = 0;
  for (auto &v : init.velocity) {
    v.x -= vx / n;
    v.y -= vy / n;
    v.z -= vz / n;
    speed2 += double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z;
  }
  double scale = std::sqrt((3.0 * n - 3.0) * c.temperature / speed2);
  // double scale = std::sqrt(3.0 * n * c.temperature / speed2);
  for (auto &v : init.velocity) {
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
  }
  return init;
}

static Params parameters(const Config &c, const InitialState &init) {
  Params p{};
  p.n = static_cast<int>(init.position.size());
  p.pitch = (p.n + 31) / 32 * 32;
  p.box = init.box;
  p.inv_box = 1.0f / p.box;
  p.dt = static_cast<float>(c.dt);
  float cutoff = static_cast<float>(c.cutoff);
  float radius = static_cast<float>(c.cutoff + c.skin);
  if (!std::isfinite(p.box) || !(p.box > 2 * cutoff) || !std::isfinite(radius))
    throw std::runtime_error("must have L > 2*cutoff");
  p.cutoff2 = cutoff * cutoff;
  p.list2 = radius * radius;
  p.rebuild2 = static_cast<float>((.49 * c.skin) * (.49 * c.skin));
  // rebuild = step % 20 == 0;
  if (!(p.dt > 0) || !std::isfinite(p.dt) || !(p.rebuild2 > 0) ||
      !std::isfinite(p.rebuild2) || !(p.cutoff2 > 0) ||
      !std::isfinite(p.list2) || !std::isfinite(p.inv_box))
    throw std::runtime_error("nan");
  int max_axis = std::max(1, static_cast<int>(std::cbrt(p.n)));
  p.nc = static_cast<int>(std::max(
      1.0, std::min(double(max_axis), std::floor(double(p.box) / radius))));
  p.cell_count = p.nc * p.nc * p.nc;
  p.inv_cell = p.nc / p.box;
  double expected = (double(p.n) / (double(p.box) * p.box * p.box)) *
                    (4.0 / 3.0) * 3.141592653589793 * radius * radius * radius;
  p.capacity =
      c.capacity
          ? c.capacity
          : static_cast<int>(std::min(
                double(p.n - 1), 32.0 * std::ceil((1.7 * expected + 32) / 32)));
  p.rebuild_every_step = c.rebuild_every_step;
  double inv2 = 1.0 / double(p.cutoff2);
  double inv6 = inv2 * inv2 * inv2;
  p.shift = 4.0 * inv6 * (inv6 - 1.0);
  // p.shift = 0.0;
  return p;
}

struct Snapshot {
  std::vector<float4> position, velocity, force;
  std::vector<int> count, neighbors;
};

class Sim {
public:
  Params p;
  int lanes;
  bool graphs;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  DeviceBuffer<float4> pos, vel, alt_pos, alt_vel, forces, displacement;
  DeviceBuffer<int> cell_ids, cell_counts, offsets, neighbors, neighbor_count;
  DeviceBuffer<Status> status;
  DeviceBuffer<DeviceState> state;
  DeviceBuffer<Totals> block_totals, total;
  DeviceBuffer<unsigned char> scan_scratch, reduce_scratch;
  size_t scan_bytes = 0, reduce_bytes = 0;

  Sim(const Config &c, const InitialState &init)
      : p(parameters(c, init)), lanes(c.lanes), graphs(c.graphs) {
    try {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      pos.allocate(p.n);
      vel.allocate(p.n);
      alt_pos.allocate(p.n);
      alt_vel.allocate(p.n);
      forces.allocate(p.n);
      displacement.allocate(p.n);
      cell_ids.allocate(p.n);
      cell_counts.allocate(p.cell_count + 1);
      offsets.allocate(p.cell_count + 1);
      neighbors.allocate(static_cast<size_t>(p.pitch) * p.capacity);
      neighbor_count.allocate(p.n);
      status.allocate(1);
      state.allocate(1);
      block_totals.allocate(blocks());
      total.allocate(1);
      Status initial_status{};
      DeviceState s{
          pos.ptr,     vel.ptr,          alt_pos.ptr,        alt_vel.ptr,
          forces.ptr,  displacement.ptr, cell_ids.ptr,       cell_counts.ptr,
          offsets.ptr, neighbors.ptr,    neighbor_count.ptr, status.ptr};
      CUDA_CHECK(cudaMemcpyAsync(pos.ptr, init.position.data(),
                                 p.n * sizeof(float4), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(vel.ptr, init.velocity.data(),
                                 p.n * sizeof(float4), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(status.ptr, &initial_status, sizeof(Status),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(state.ptr, &s, sizeof(s),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(
          neighbors.ptr, 0xff,
          static_cast<size_t>(p.pitch) * p.capacity * sizeof(int), stream));
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
                                               cell_counts.ptr, offsets.ptr,
                                               p.cell_count + 1, stream));
      CUDA_CHECK(cub::DeviceReduce::Reduce(
          nullptr, reduce_bytes, block_totals.ptr, total.ptr, blocks(),
          AddTotals{}, Totals{}, stream));
      scan_scratch.allocate(scan_bytes);
      reduce_scratch.allocate(reduce_bytes);
      rebuild();
      get_status();
      launch_force(0.0f);
      get_status();
      if (graphs)
        make_graph();
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      if (graphs)
        CUDA_CHECK(cudaGraphUpload(executable, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
      release();
      throw;
    }
  }

  Sim(const Sim &) = delete;
  ~Sim() { release(); }

  int blocks() const { return (p.n + kBlock - 1) / kBlock; }
  int force_blocks() const { return (p.n * lanes + kBlock - 1) / kBlock; }

  void release() noexcept {
    if (stream)
      cudaStreamSynchronize(stream);
    if (executable)
      cudaGraphExecDestroy(executable);
    if (graph)
      cudaGraphDestroy(graph);
    if (start)
      cudaEventDestroy(start);
    if (stop)
      cudaEventDestroy(stop);
    if (stream)
      cudaStreamDestroy(stream);
    stream = nullptr;
    executable = nullptr;
    graph = nullptr;
    start = stop = nullptr;
  }

  const void *force_function() const {
    switch (lanes) {
    case 1:
      return reinterpret_cast<const void *>(force_kick<1>);
    case 2:
      return reinterpret_cast<const void *>(force_kick<2>);
    case 4:
      return reinterpret_cast<const void *>(force_kick<4>);
    default:
      return reinterpret_cast<const void *>(force_kick<8>);
    }
  }

  void launch_force(float kick) {
    void *args[] = {&state.ptr, &p, &kick};
    CUDA_CHECK(cudaLaunchKernel(force_function(), dim3(force_blocks()),
                                dim3(kBlock), args, 0, stream));
  }

  void rebuild() {
    // offsets[0] = 0;
    // for (int c = 0; c < p.cell_count; c++)
    //   offsets[c + 1] = offsets[c] + counts[c];
    CUDA_CHECK(cudaMemsetAsync(cell_counts.ptr, 0,
                               (p.cell_count + 1) * sizeof(int), stream));
    count_cells<<<blocks(), kBlock, 0, stream>>>(state.ptr, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_scratch.ptr, scan_bytes,
                                             cell_counts.ptr, offsets.ptr,
                                             p.cell_count + 1, stream));
    CUDA_CHECK(cudaMemsetAsync(cell_counts.ptr, 0,
                               (p.cell_count + 1) * sizeof(int), stream));
    scatter_particles<<<blocks(), kBlock, 0, stream>>>(state.ptr, p);
    swap_particles<<<1, 1, 0, stream>>>(state.ptr);
    build_neighbors<<<blocks(), kBlock, 0, stream>>>(state.ptr, p);
    finish_build<<<1, 1, 0, stream>>>(state.ptr);
    CUDA_CHECK(cudaGetLastError());
  }

  cudaGraphNode_t kernel_node(const void *function, int grid, int block,
                              void **args,
                              cudaGraphNode_t dependency = nullptr) {
    cudaKernelNodeParams parameters{};
    parameters.func = const_cast<void *>(function);
    parameters.gridDim = dim3(grid);
    parameters.blockDim = dim3(block);
    parameters.kernelParams = args;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph,
                                      dependency ? &dependency : nullptr,
                                      dependency ? 1 : 0, &parameters));
    return node;
  }

  void make_graph() {
    // drift<<<blocks(), kBlock, 0, stream>>>(state.ptr, p);
    // CUDA_CHECK(cudaGetLastError());
    // if (get_status().rebuild) rebuild();
    // launch_force(0.5f * p.dt);
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                                cudaGraphCondAssignDefault));
    void *drift_args[] = {&state.ptr, &p};
    auto first = kernel_node(reinterpret_cast<const void *>(drift), blocks(),
                             kBlock, drift_args);
    void *choose_args[] = {&state.ptr, &handle};
    auto choose = kernel_node(reinterpret_cast<const void *>(choose_rebuild), 1,
                              1, choose_args, first);
    cudaGraphNodeParams conditional{};
    conditional.type = cudaGraphNodeTypeConditional;
    conditional.conditional.handle = handle;
    conditional.conditional.type = cudaGraphCondTypeIf;
    conditional.conditional.size = 1;
    cudaGraphNode_t branch;
    CUDA_CHECK(cudaGraphAddNode(&branch, graph, &choose, 1, &conditional));
    cudaGraph_t body = conditional.conditional.phGraph_out[0];
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeRelaxed));
    rebuild();
    CUDA_CHECK(cudaStreamEndCapture(stream, nullptr));
    float kick = 0.5f * p.dt;
    void *force_args[] = {&state.ptr, &p, &kick};
    kernel_node(force_function(), force_blocks(), kBlock, force_args, branch);
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  }

  void check_status(const Status &s) const {
    if (s.error & kOverflow)
      throw std::runtime_error("list overflow");
    if (s.error & kOverlap)
      throw std::runtime_error("overlap");
    if (s.error & kNonfinite)
      throw std::runtime_error("nan state");
  }

  Status get_status() {
    Status s;
    CUDA_CHECK(cudaMemcpyAsync(&s, status.ptr, sizeof(s),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_status(s);
    return s;
  }

  double advance(int steps) {
    // auto t0 = std::chrono::steady_clock::now();
    // cudaGraphLaunch(executable, stream);
    // auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < steps; i++) {
      if (graphs) {
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
      } else {
        drift<<<blocks(), kBlock, 0, stream>>>(state.ptr, p);
        CUDA_CHECK(cudaGetLastError());
        if (get_status().rebuild)
          rebuild();
        launch_force(0.5f * p.dt);
      }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    // CUDA_CHECK(cudaDeviceSynchronize());
    get_status();
    return ms;
  }

  Totals measure() {
    observe<<<blocks(), kBlock, 0, stream>>>(state.ptr, p, block_totals.ptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceReduce::Reduce(reduce_scratch.ptr, reduce_bytes,
                                         block_totals.ptr, total.ptr, blocks(),
                                         AddTotals{}, Totals{}, stream));
    Totals result;
    Status s;
    CUDA_CHECK(cudaMemcpyAsync(&result, total.ptr, sizeof(result),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&s, status.ptr, sizeof(s),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_status(s);
    return result;
  }

  Snapshot snapshot() {
    // auto position = pos.ptr;
    DeviceState s;
    CUDA_CHECK(cudaMemcpyAsync(&s, state.ptr, sizeof(s), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Snapshot result;
    result.position.resize(p.n);
    result.velocity.resize(p.n);
    result.force.resize(p.n);
    result.count.resize(p.n);
    result.neighbors.resize(static_cast<size_t>(p.pitch) * p.capacity);
    CUDA_CHECK(cudaMemcpyAsync(result.position.data(), s.position,
                               p.n * sizeof(float4), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(result.velocity.data(), s.velocity,
                               p.n * sizeof(float4), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(result.force.data(), forces.ptr,
                               p.n * sizeof(float4), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(result.count.data(), neighbor_count.ptr,
                               p.n * sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(result.neighbors.data(), neighbors.ptr,
                               result.neighbors.size() * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
  }

  void write_xyz(std::ostream &out, long long step,
                 std::vector<float4> &positions,
                 std::vector<float4> &velocities) {
    DeviceState s;
    CUDA_CHECK(cudaMemcpyAsync(&s, state.ptr, sizeof(s), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    positions.resize(p.n);
    velocities.resize(p.n);
    CUDA_CHECK(cudaMemcpyAsync(positions.data(), s.position,
                               p.n * sizeof(float4), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(velocities.data(), s.velocity,
                               p.n * sizeof(float4), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    out << p.n << '\n'
        << std::setprecision(9) << "Lattice=\"" << p.box << " 0 0 0 " << p.box
        << " 0 0 0 " << p.box << "\" Origin=\"0 0 0\" pbc=\"T T T\""
        << " Properties=id:I:1:type:I:1:pos:R:3:velo:R:3"
        << " Step=" << step << std::setprecision(17)
        << " Time=" << step * double(p.dt) << '\n'
        << std::setprecision(9);
    for (int i = 0; i < p.n; i++) {
      auto r = positions[i], v = velocities[i];
      out << static_cast<int>(r.w) + 1 << " 1 " << r.x << ' ' << r.y << ' '
          << r.z << ' ' << v.x << ' ' << v.y << ' ' << v.z << '\n';
    }
    out.flush();
  }
};

struct Vec3 {
  double x = 0, y = 0, z = 0;
};

struct Reference {
  std::vector<Vec3> force;
  double potential = 0, virial = 0;
};

static Reference reference(const std::vector<float4> &position,
                           const Params &p) {
  Reference ref;
  ref.force.resize(p.n);
  auto minimum_image = [&](double x) {
    return x - p.box * std::nearbyint(x / p.box);
  };
  for (int i = 0; i < p.n; i++)
    for (int j = i + 1; j < p.n; j++) {
      double dx = minimum_image(double(position[i].x) - position[j].x);
      double dy = minimum_image(double(position[i].y) - position[j].y);
      double dz = minimum_image(double(position[i].z) - position[j].z);
      double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= p.cutoff2)
        continue;
      if (r2 == 0)
        throw std::runtime_error("overlap in CPU reference");
      double inv2 = 1.0 / r2, inv6 = inv2 * inv2 * inv2;
      double scale = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0);
      ref.force[i].x += scale * dx;
      ref.force[j].x -= scale * dx;
      ref.force[i].y += scale * dy;
      ref.force[j].y -= scale * dy;
      ref.force[i].z += scale * dz;
      ref.force[j].z -= scale * dz;
      ref.potential += 4.0 * inv6 * (inv6 - 1.0) - p.shift;
      ref.virial += scale * r2;
    }
  return ref;
}

static double verify(Sim &sim) {
  Snapshot s = sim.snapshot();
  Reference ref = reference(s.position, sim.p);
  double worst = 0;
  std::vector<int> ids(sim.p.n), seen(sim.p.n, -1);
  for (int i = 0; i < sim.p.n; i++) {
    auto r = s.position[i];
    if (!(r.x >= 0 && r.x < sim.p.box && r.y >= 0 && r.y < sim.p.box &&
          r.z >= 0 && r.z < sim.p.box))
      throw std::runtime_error("periodic wrapping failed");
    int id = static_cast<int>(r.w);
    if (id < 0 || id >= sim.p.n || ids[id]++)
      throw std::runtime_error("particle permutation id lost");
    for (int k = 0; k < s.count[i]; k++) {
      int j = s.neighbors[static_cast<size_t>(k) * sim.p.pitch + i];
      if (j < 0 || j >= sim.p.n || j == i || seen[j] == i)
        throw std::runtime_error("duplicate neighbor index");
      seen[j] = i;
    }
    for (int j = 0; j < sim.p.n; j++) {
      if (i == j)
        continue;
      double dx = double(r.x) - s.position[j].x;
      double dy = double(r.y) - s.position[j].y;
      double dz = double(r.z) - s.position[j].z;
      dx -= sim.p.box * std::nearbyint(dx / sim.p.box);
      dy -= sim.p.box * std::nearbyint(dy / sim.p.box);
      dz -= sim.p.box * std::nearbyint(dz / sim.p.box);
      if (dx * dx + dy * dy + dz * dz < sim.p.cutoff2 && seen[j] != i)
        throw std::runtime_error("verlet list omitted an interacting pair");
    }
    double a[3] = {s.force[i].x, s.force[i].y, s.force[i].z};
    double b[3] = {ref.force[i].x, ref.force[i].y, ref.force[i].z};
    for (int axis = 0; axis < 3; axis++) {
      double error = std::abs(a[axis] - b[axis]) / (1.0 + std::abs(b[axis]));
      if (!std::isfinite(error))
        throw std::runtime_error("nan force");
      worst = std::max(worst, error);
    }
  }
  if (worst > 2e-4)
    throw std::runtime_error("force mismatch: " + std::to_string(worst));
  Totals t = sim.measure();
  if (std::abs(t.potential - ref.potential) >
          2e-5 * (1 + std::abs(ref.potential)) ||
      std::abs(t.virial - ref.virial) > 5e-5 * (1 + std::abs(ref.virial)))
    throw std::runtime_error("energy or virial mismatch");
  return worst;
}

static double energy(const Totals &t) { return t.kinetic + t.potential; }

static void verify_verlet_step(Sim &sim, const InitialState &init) {
  auto positions = init.position;
  auto old = reference(positions, sim.p);
  std::vector<Vec3> velocity(sim.p.n);
  double dt = sim.p.dt;
  auto periodic = [&](double x) {
    x -= sim.p.box * std::floor(x / sim.p.box);
    float r = static_cast<float>(x);
    return r >= sim.p.box ? 0.0f : r;
  };
  for (int i = 0; i < sim.p.n; i++) {
    velocity[i] = {init.velocity[i].x + .5 * dt * old.force[i].x,
                   init.velocity[i].y + .5 * dt * old.force[i].y,
                   init.velocity[i].z + .5 * dt * old.force[i].z};
    positions[i].x = periodic(positions[i].x + dt * velocity[i].x);
    positions[i].y = periodic(positions[i].y + dt * velocity[i].y);
    positions[i].z = periodic(positions[i].z + dt * velocity[i].z);
  }
  auto next = reference(positions, sim.p);
  sim.advance(1);
  auto actual = sim.snapshot();
  for (int i = 0; i < sim.p.n; i++) {
    int id = static_cast<int>(actual.position[i].w);
    double position_error[] = {double(actual.position[i].x) - positions[id].x,
                               double(actual.position[i].y) - positions[id].y,
                               double(actual.position[i].z) - positions[id].z};
    double velocity_error[] = {
        actual.velocity[i].x - (velocity[id].x + .5 * dt * next.force[id].x),
        actual.velocity[i].y - (velocity[id].y + .5 * dt * next.force[id].y),
        actual.velocity[i].z - (velocity[id].z + .5 * dt * next.force[id].z)};
    for (int axis = 0; axis < 3; axis++) {
      double d = position_error[axis];
      d -= sim.p.box * std::nearbyint(d / sim.p.box);
      if (std::abs(d) > 2e-5 || std::abs(velocity_error[axis]) > 3e-5)
        throw std::runtime_error("verlet step mismatch");
    }
  }
}

static void testt(bool graphs) {
  double worst = 0;
  for (double distance : {1.0, std::pow(2.0, 1.0 / 6.0), 2.49, 2.51}) {
    Config c;
    c.graphs = graphs;
    c.lanes = 4;
    InitialState init{8.0f,
                      {make_float4(.25f, 1, 1, 0),
                       make_float4(float(8.25 - distance), 1, 1, 1)},
                      {make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)}};
    Sim sim(c, init);
    worst = std::max(worst, verify(sim));
    if (distance == 1.0) {
      auto s = sim.snapshot();
      for (int i = 0; i < 2; i++) {
        double expected = s.position[i].w == 0 ? 24 : -24;
        if (std::abs(s.force[i].x - expected) > 1e-4)
          throw std::runtime_error("sign check failed");
      }
    }
    verify_verlet_step(sim, init);
  }
  for (int lanes : {1, 2, 4, 8}) {
    Config c;
    c.graphs = graphs;
    c.cells = 4;
    c.lanes = lanes;
    c.skin = .12;
    auto init = initialize(c, true);
    Sim sim(c, init);
    worst = std::max(worst, verify(sim));
    verify_verlet_step(sim, init);
    for (int steps : {1, 7, 31}) {
      sim.advance(steps);
      worst = std::max(worst, verify(sim));
    }
    if (sim.get_status().builds <= 1)
      throw std::runtime_error("rebuild fail");
    std::cout << "  lanes=" << lanes << ": pass\n";
  }
  for (double density : {.8, .15}) {
    Config c;
    c.graphs = graphs;
    c.cells = 5;
    c.density = density;
    c.skin = .12;
    auto init = initialize(c, true);
    Sim sim(c, init);
    verify_verlet_step(sim, init);
    sim.advance(37);
    worst = std::max(worst, verify(sim));
  }
  {
    Config c;
    c.graphs = graphs;
    InitialState init{8.0f,
                      {make_float4(7.999f, 1, 1, 0), make_float4(4, 1, 1, 1)},
                      {make_float4(1, 0, 0, 0), make_float4(-1, 0, 0, 0)}};
    Sim sim(c, init);
    verify_verlet_step(sim, init);
    worst = std::max(worst, verify(sim));
  }
  Config c;
  c.graphs = graphs;
  c.cells = 4;
  c.skin = .12;
  auto init = initialize(c, true);
  Snapshot a, b;
  {
    Sim adaptive(c, init);
    adaptive.advance(20);
    a = adaptive.snapshot();
  }
  {
    c.rebuild_every_step = true;
    Sim always(c, init);
    always.advance(20);
    b = always.snapshot();
    if (always.get_status().builds != 21)
      throw std::runtime_error("fail");
  }
  std::vector<float4> by_id(a.position.size());
  for (auto r : a.position)
    by_id[static_cast<int>(r.w)] = r;
  for (auto r : b.position) {
    auto q = by_id[static_cast<int>(r.w)];
    for (double d : {double(r.x) - q.x, double(r.y) - q.y, double(r.z) - q.z}) {
      d -= init.box * std::nearbyint(d / init.box);
      if (std::abs(d) > 2e-4)
        throw std::runtime_error("fail");
    }
  }
  for (double dt : {.002, .001}) {
    Config e;
    e.graphs = graphs;
    e.cells = 4;
    e.temperature = .5;
    e.dt = dt;
    Sim sim(e, initialize(e, true));
    double e0 = energy(sim.measure()), max_drift = 0;
    int steps = static_cast<int>(std::lround(.8 / dt));
    for (int done = 0; done < steps; done += 100) {
      sim.advance(100);
      max_drift =
          std::max(max_drift, std::abs(energy(sim.measure()) - e0) / sim.p.n);
    }
    worst = std::max(worst, verify(sim));
    if (max_drift > .002)
      throw std::runtime_error("fail");
    std::cout << "  dt=" << dt << ": max |delta E|/N = " << max_drift << '\n';
  }
  bool overflow_detected = false;
  try {
    Config tiny;
    tiny.graphs = graphs;
    tiny.cells = 4;
    tiny.capacity = 1;
    Sim sim(tiny, initialize(tiny));
  } catch (const std::runtime_error &e) {
    overflow_detected =
        std::string(e.what()).find("overflow") != std::string::npos;
    if (!overflow_detected)
      throw;
  }
  if (!overflow_detected)
    throw std::runtime_error("not overflow");
  std::cout << "pass\n"
               "max error: "
            << worst << '\n';
}

static Config parse(int argc, char **argv) {
  Config c;
  for (int i = 1; i < argc; i++) {
    std::string option = argv[i];
    if (option == "--check") {
      c.check = true;
      continue;
    }
    if (option == "--no-graphs") {
      c.graphs = false;
      continue;
    }
    if (option == "--rebuild-every-step") {
      c.rebuild_every_step = true;
      continue;
    }
    i++;
    if (i == argc)
      throw std::runtime_error("missing value for " + option);
    std::string value = argv[i];
    auto real = [&] {
      size_t end;
      double x = std::stod(value, &end);
      if (end != value.size() || !std::isfinite(x))
        throw std::runtime_error("invalid value for " + option);
      return x;
    };
    auto integer = [&] {
      double x = real();
      if (x != std::floor(x) || x < 0 || x > std::numeric_limits<int>::max())
        throw std::runtime_error("integer must be >= 0 for " + option);
      return static_cast<int>(x);
    };
    if (option == "--cells")
      c.cells = integer();
    else if (option == "--steps")
      c.steps = integer();
    else if (option == "--warmup")
      c.warmup = integer();
    else if (option == "--thermo")
      c.thermo = integer();
    else if (option == "--dump")
      c.dump = integer();
    else if (option == "--output")
      c.output = value;
    else if (option == "--neighbors")
      c.capacity = integer();
    else if (option == "--lanes")
      c.lanes = integer();
    else if (option == "--seed")
      c.seed = integer();
    else if (option == "--density")
      c.density = real();
    else if (option == "--temperature")
      c.temperature = real();
    else if (option == "--dt")
      c.dt = real();
    else if (option == "--cutoff")
      c.cutoff = real();
    else if (option == "--skin")
      c.skin = real();
    else
      throw std::runtime_error("unknown option: " + option);
  }
  if (c.cells < 1 || c.cells > 128 || c.density <= 0 || c.temperature < 0 ||
      c.dt <= 0 || c.cutoff <= 0 || c.skin <= 0 ||
      (c.lanes != 1 && c.lanes != 2 && c.lanes != 4 && c.lanes != 8))
    throw std::runtime_error("invalid parameters");
  return c;
}

static void print_thermo(int step, const Totals &t, const Sim &sim,
                         double baseline, const Status &status) {
  double com_ke = (t.px * t.px + t.py * t.py + t.pz * t.pz) / (2 * sim.p.n);
  double thermal_ke = t.kinetic - com_ke;
  double temperature = 2 * thermal_ke / (3 * sim.p.n - 3);
  double volume = double(sim.p.box) * sim.p.box * sim.p.box;
  double pressure = (2 * thermal_ke + t.virial) / (3 * volume);
  // double temperature = 2 * t.kinetic / (3 * sim.p.n);
  std::cout << std::setw(9) << step << std::fixed << std::setprecision(6)
            << std::setw(12) << temperature << std::setw(13)
            << t.potential / sim.p.n << std::setw(13) << energy(t) / sim.p.n
            << std::setw(13) << (energy(t) - baseline) / sim.p.n
            << std::setw(12) << pressure << std::setw(10)
            << t.neighbors / sim.p.n << std::setw(10) << status.builds << '\n';
}

int main(int argc, char **argv) {
  try {
    Config c = parse(argc, argv);
    // c.cells = 4;
    // c.steps = 100;
    // c.warmup = 0;
    // c.thermo = 10;
    // c.rebuild_every_step = true;
    cudaDeviceProp gpu{};
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    std::cout << "GPU: " << gpu.name << " (sm_" << gpu.major << gpu.minor
              << ")\n";
    if (c.check) {
      testt(c.graphs);
      return 0;
    }
    // {
    //   Config test;
    //   test.cells = 4;
    //   test.skin = .08;
    //   auto init = initialize(test, true);
    //   Sim small(test, init);
    //   verify_verlet_step(small, init);
    //   for (int step = 0; step < 100; step++) {
    //     small.advance(1);
    //     std::cout << step << " force error " << verify(small) << '\n';
    //   }
    //   return 0;
    // }
    // {
    //   Config test;
    //   test.cells = 4;
    //   auto init = initialize(test, true);
    //   for (double dt : {.002, .001}) {
    //     test.dt = dt;
    //     Sim small(test, init);
    //     double e0 = energy(small.measure());
    //     small.advance(int(std::lround(1.0 / dt)));
    //     std::cout << dt << " dE/N "
    //               << (energy(small.measure()) - e0) / small.p.n << '\n';
    //   }
    //   return 0;
    // }
    // {
    //   Config test;
    //   test.dt = 1;
    //   InitialState init{
    //       8,
    //       {make_float4(.25f, 1, 1, 0), make_float4(4.25f, 1, 1, 1)},
    //       {make_float4(17, 0, 0, 0), make_float4(17, 0, 0, 0)}};
    //   Sim small(test, init);
    //   verify_verlet_step(small, init);
    //   std::cout << "periodic error " << verify(small) << '\n';
    //   return 0;
    // }

    Sim sim(c, initialize(c));
    std::ofstream trajectory;
    std::vector<float4> dump_positions, dump_velocities;
    if (c.dump) {
      trajectory.imbue(std::locale::classic());
      trajectory.open(c.output);
      if (!trajectory)
        throw std::runtime_error("cannot open trajectory: " + c.output);
      trajectory.exceptions(std::ios::failbit | std::ios::badbit);
      std::cout << "OVITO: " << c.output << " every " << c.dump << "\n";
    }
    std::cout << "N=" << sim.p.n << "  L=" << sim.p.box << "  rho=" << c.density
              << "  cutoff=" << c.cutoff << "  skin=" << c.skin
              << "  dt=" << c.dt << "  lanes=" << c.lanes
              << "  neighbor capacity=" << sim.p.capacity
              << "  grid=" << sim.p.nc << '^' << 3 << "  graphs=" << c.graphs
              << '\n';
    if (c.warmup) {
      std::cout << "warmup: " << c.warmup << " steps\n";
      sim.advance(c.warmup);
    }
    // {
    //   auto t = sim.measure();
    //   std::cout << "P = " << t.px << ' ' << t.py << ' ' << t.pz << '\n';
    //   std::cout << "2K/(3N-3) = " << 2 * t.kinetic / (3 * sim.p.n - 3) <<
    //   '\n';
    // }
    // {
    //   auto initial = sim.snapshot();
    //   std::vector<float4> v0(sim.p.n);
    //   for (int i = 0; i < sim.p.n; i++)
    //     v0[int(initial.position[i].w)] = initial.velocity[i];
    //   for (int step = 10; step <= 100; step += 10) {
    //     sim.advance(10);
    //     auto s = sim.snapshot();
    //     double correlation = 0;
    //     for (int i = 0; i < sim.p.n; i++) {
    //       auto a = v0[int(s.position[i].w)], b = s.velocity[i];
    //       correlation +=
    //           double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
    //     }
    //     std::cout << step * sim.p.dt << ' ' << correlation / sim.p.n << '\n';
    //   }
    //   return 0;
    // }

    Totals first = sim.measure();
    double baseline = energy(first);
    auto initial_status = sim.get_status();
    std::cout << "     step           T          U/N          E/N         dE/N "
                 "          P    nbrs/N    builds\n";
    print_thermo(0, first, sim, baseline, initial_status);
    if (c.dump)
      sim.write_xyz(trajectory, c.warmup, dump_positions, dump_velocities);
    double gpu_ms = 0;
    int done = 0;
    auto wall_start = std::chrono::steady_clock::now();
    while (done < c.steps) {
      int count = c.steps - done;
      if (c.thermo)
        count = std::min(count, c.thermo - done % c.thermo);
      if (c.dump)
        count = std::min(count, c.dump - done % c.dump);
      gpu_ms += sim.advance(count);
      done += count;
      // auto s = sim.snapshot();
      // std::cout << s.position[0].x << ' ' << s.velocity[0].x << '\n';
      if (done == c.steps || (c.thermo && done % c.thermo == 0)) {
        Totals t = sim.measure();
        print_thermo(done, t, sim, baseline, sim.get_status());
      }
      if (c.dump && (done == c.steps || done % c.dump == 0))
        sim.write_xyz(trajectory, static_cast<long long>(c.warmup) + done,
                      dump_positions, dump_velocities);
    }
    if (c.dump)
      trajectory.close();
    double t = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - wall_start)
                   .count();
    auto final_status = sim.get_status();
    if (c.steps) {
      std::cout << std::setprecision(3)
                << "event timestep time: " << gpu_ms / c.steps << " ms/step, "
                << double(sim.p.n) * c.steps / (gpu_ms * 1000)
                << " M particle-steps/s\n"
                << "time: " << t / c.steps << " ms/step\n"
                << "rebuilds: " << final_status.builds - initial_status.builds
                << "; max neighbors seen: " << final_status.max_neighbors
                << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
