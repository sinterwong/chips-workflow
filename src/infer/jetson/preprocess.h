#ifndef __INFER_PREPROCESS_H
#define __INFER_PREPROCESS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace infer {
namespace trt {
struct AffineMatrix {
  float value[6];
};

void preprocess_kernel_img(uint8_t *src, int src_width, int src_height,
                           float *dst, int dst_width, int dst_height,
                           float alpha, float beta, cudaStream_t stream);
} // namespace trt
} // namespace infer

#endif // __INFER_PREPROCESS_H
