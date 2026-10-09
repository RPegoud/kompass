#include "../common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

inline void check_gemm_inputs(const at::Tensor &X, const at::Tensor &Y) {
  TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(X.device() == Y.device(), "Inputs must be on the same device")
  TORCH_CHECK(X.is_contiguous() && Y.is_contiguous(),
              "Inputs must be contiguous");
  //   TODO: add shape check
}

__global__ void GemmFwdKernel(float *X, float *Y, float *Z, int M, int N, int K) {
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

