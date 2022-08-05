#ifndef __INFER_PREPROCESS_STRANDARD_H
#define __INFER_PREPROCESS_STRANDARD_H

#include <cstdint>
#include <cuda_runtime.h>

namespace infer {
namespace trt {

void strandard_image(uint8_t *src, float *dst, int size,
                     float alpha, float beta, cudaStream_t stream);
} // namespace trt
} // namespace infer

#endif // __INFER_PREPROCESS_H
