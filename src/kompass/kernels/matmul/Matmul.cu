#include "../common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

inline void check_matmul_inputs(const at::Tensor &X, const at::Tensor &Y) {
  TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(X.device() == Y.device(), "Inputs must be on the same device")
  TORCH_CHECK(X.is_contiguous() && Y.is_contiguous(),
              "Inputs must be contiguous");
}

__global__ void naiveMatmulFwdKernel(float *X, float *Y, float *Z, int M, int N,
                                     int K) {
  // X is [M, K], Y is [K, N]
  int row = blockDim.y * blockIdx.y + threadIdx.y; // matrix Z row (0 to M-1)
  int col = blockDim.x * blockIdx.x + threadIdx.x; // matrix Z col (0 to N-1)

  if ((row < M) && (col < N)) { // Z is [M, N]
    float acc = 0.0;
    for (int k = 0; k < K; k++) { // iterate over shared dimension
      acc += X[row * K + k] * Y[N * k + col];
    }
    Z[row * N + col] = acc;
  }
}

#define TILE_WIDTH 32 // global variable
__global__ void tiledMatmulFwdKernel(float *X, float *Y, float *Z, int Width) {
  // define tiles in shared memory
  __shared__ float Xds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Yds[TILE_WIDTH][TILE_WIDTH];

  // get block and thread indices
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int ty = threadIdx.y;

  // identify which tile of Z to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float acc = 0.0;
  for (int b = 0; b < Width / TILE_WIDTH; b++) {
    // colaboratively load X and Y tiles into shared memory
    Xds[ty][tx] = X[Row * Width + b * TILE_WIDTH + tx];
    Yds[ty][tx] = Y[Col + (b * TILE_WIDTH + ty) * Width];
    __syncthreads(); // wait until all threads have loaded the tiles

    // accumulate dot products over the current tile
    // along the K dimension
    for (int k = 0; k < TILE_WIDTH; k++) {
      acc += Xds[ty][k] * Yds[k][tx];
    }
    __syncthreads(); // wait until all threads have finished using the tile
  }
  Z[Row * Width + Col] = acc;
}

at::Tensor tiled_matmul_fwd(const at::Tensor &X, const at::Tensor &Y) {
  // TODO: remove the square matrix constraint
  // TODO: implement register tiling
  
  check_matmul_inputs(X, Y);
  const at::cuda::CUDAGuard guard(X.device());

  auto Z = at::empty_like(X); // TODO: initialise with [M,N]
  at::IntArrayRef MK = X.sizes();
  at::IntArrayRef KN = Y.sizes();
  int Width = MK[0];

  int N_BLOCKS = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 n_blocks(N_BLOCKS, N_BLOCKS);
  dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
  tiledMatmulFwdKernel<<<n_blocks, threads_per_block>>>(
      X.data_ptr<float>(), Y.data_ptr<float>(), Z.data_ptr<float>(), Width);

  CUDA_CHECK(
      cudaGetLastError()); // catch config errors (e.g. invalid block size)

  return Z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tiled_cuda", &tiled_matmul_fwd);
}