#include "jetson/standard.h"
#include <opencv2/opencv.hpp>

namespace infer {
namespace trt {
__global__ void standard_kernel(uint8_t *src, float *dst, float alpha,
                                  float beta, int edge) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;

  if (alpha != 0.0) {
    dst[position] = src[position] * 1.0 / alpha;
  } else {
    dst[position] = src[position] * 1.0;
  }
  if (beta != 0.0) {
    dst[position] = dst[position] - beta;
  }
}

void strandard_image(uint8_t *src, float *dst, int size,
              float alpha, float beta, cudaStream_t stream) {

  int threads = 256;
  int blocks = ceil(size / (float)threads);
  standard_kernel<<<blocks, threads, 0, stream>>>(src, dst, alpha, beta, size);
  cudaStreamSynchronize(stream);
}

} // namespace trt
} // namespace infer
