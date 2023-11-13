/**
 * @file test_faiss.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <iostream>
#include <vector>

void normalize_L2(float *x, int d) {
  float sum = 0;
  for (int i = 0; i < d; i++) {
    sum += x[i] * x[i];
  }
  sum = std::sqrt(sum);
  for (int i = 0; i < d; i++) {
    x[i] /= sum;
  }
}

int main() {
  // Initialize random seed
  srand(123);

  // Number of vectors and vector dimensionality
  int nb = 100000;
  int d = 512;

  // Randomly generate vecotors and normalize them
  float *xb = new float[d * nb];
  for (int i = 0; i < nb * d; i++) {
    xb[i] = rand() % 1000;
  }

  for (int i = 0; i < nb; i++) {
    normalize_L2(xb + i * d, d);
  }

  // Generate a random query vector and normalize it
  float *xq = new float[d];
  for (int i = 0; i < d; i++) {
    xq[i] = rand() % 1000;
  }

  normalize_L2(xq, d);

  // Build and index for inner product(dot product)
  faiss::IndexFlatIP flatIndex(d);
  faiss::IndexIDMap2 index(&flatIndex);

  // faiss::idx_t ids_to_add[] = {0, 1, 4, 6, 8};
  std::vector<long> ids_to_add;
  for (int i = 0; i < nb; ++i) {
    ids_to_add.push_back(i);
  }
  index.add_with_ids(nb, xb, ids_to_add.data());

  faiss::idx_t new_id_to_add[] = {50};
  index.add_with_ids(1, xq, new_id_to_add);

  // Search for the top 2 most similar vectors
  int k = 2;
  float *distances = new float[k];
  int64_t *indices = new int64_t[k];

  auto start = std::chrono::high_resolution_clock::now();
  index.search(1, xq, k, distances, indices);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  auto cost = static_cast<double>(duration.count()) / 1000;

  std::cout << "Cost time: " << cost
            << "ms, Search results (cosine similarity):" << std::endl;
  for (int i = 0; i < k; i++) {
    printf("%2d: [idx: %ld, cosine similarity: %f]\n", i, indices[i],
           distances[i]);
  }

  // Clean up
  delete[] xb;
  delete[] xq;
  delete[] distances;
  delete[] indices;

  return 0;
}