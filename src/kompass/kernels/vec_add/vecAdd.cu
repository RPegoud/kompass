#include "../common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

inline void check_vec_add_inputs(const at::Tensor &A, const at::Tensor &B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(A.device() == B.device(), "Inputs must be on the same device")
  TORCH_CHECK(A.scalar_type() == at::kFloat, "float32 only");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
              "Inputs must be contiguous");
  TORCH_CHECK(A.numel() == B.numel(), "Inputs must have the same size");
}

__global__ void vecAddFwdKernel(float *A, float *B, float *C, int n) {
  // every thread performs one pair-wise addition
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

at::Tensor vec_add_fwd(const at::Tensor &A, const at::Tensor &B) {
  check_vec_add_inputs(A, B);
  const at::cuda::CUDAGuard guard(
      A.device()); // reassigns the device variable once out of scope

  auto C = at::empty_like(A);

  const int n = A.numel();
  const int threads = 256;

  // C++ truncates towards zero, add `threads -1` before dividing to ensure we
  // get the correct result, besides integer division is more precise and faster
  // than float div e.g. `ceil(n/256.0)`
  vecAddFwdKernel<<<(n + threads - 1) / threads, threads>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);

  CUDA_CHECK(
      cudaGetLastError()); // catch config errors (e.g. invalid block size)

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vec_add_cuda", &vec_add_fwd);
}