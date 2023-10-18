/**
 * @file app_face.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
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

  // Number of vectos adn vector dimensionality
  int nb = 10;
  int d = 8;

  // Randomly generate vecotors and normalize them
  float *xb = new float[d * nb];
  for (int i = 0; i < nb * d; i++) {
    xb[i] = rand() % 1000;
  }

  for (int i = 0; i < nb; i++) {
    normalize_L2(xb + i * d, d);
  }

  // Build and index for inner product(dot product)
  faiss::IndexFlatIP index(d);

  index.add(nb, xb);

  // Print all vectors
  for (int i = 0; i < index.ntotal; i++) {
    std::cout << "Vector " << i << ":\n";
    for (int j = 0; j < d; j++) {
      std::cout << index.get_xb()[i * d + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "************************************" << std::endl;

  // Generate a random query vector and normalize it
  float *xq = new float[d];
  for (int i = 0; i < d; i++) {
    xq[i] = rand() % 1000;
  }

  normalize_L2(xq, d);

  // Remove the ID from the index
  faiss::idx_t ids_to_remove[] = {2};
  size_t ids_length = 1;
  faiss::IDSelectorArray selector{ids_length, ids_to_remove};
  index.remove_ids(selector);

  // 故意添加一个一模一样的特征
  // index.add_with_ids(idx_t n, const float *x, const idx_t *xids)

  // Print all vectors
  for (int i = 0; i < index.ntotal; i++) {
    std::cout << "Vector " << i << ":\n";
    for (int j = 0; j < d; j++) {
      std::cout << index.get_xb()[i * d + j] << " ";
    }
    std::cout << "\n";
  }

  // Search for the top 5 most similar vectors
  int k = 2;
  float *distances = new float[k];
  int64_t *indices = new int64_t[k];

  index.search(1, xq, k, distances, indices);

  std::cout << "Search results (cosine similarity):" << std::endl;
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