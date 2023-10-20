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
  int nb = 5;
  int d = 8;

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

  // index.add(nb, xb);  // IDMap不能用，暂时不清楚原因

  faiss::idx_t ids_to_add[] = {0, 1, 4, 6, 8};
  index.add_with_ids(nb, xb, ids_to_add);

  faiss::idx_t new_id_to_add[] = {50};
  index.add_with_ids(1, xq, new_id_to_add);

  // Print all vectors
  std::vector<float> tempV1(d);
  for (int i = 0; i < index.ntotal; i++) {
    faiss::idx_t id = index.id_map.at(i);
    // retrieve the vector given its ID
    index.reconstruct(id, tempV1.data());
    // print ID
    std::cout << "Vector ID: " << id << "\n";
    // print vector components
    std::cout << "Components: [";
    for (int j = 0; j < d; j++) {
      std::cout << tempV1[j];
      if (j != d - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "*********************************" << std::endl;

  // Remove the ID from the index
  faiss::idx_t ids_to_remove[] = {4, 6};
  size_t ids_length = 2;
  faiss::IDSelectorArray selector{ids_length, ids_to_remove};
  index.remove_ids(selector);

  // Print all vectors
  std::vector<float> tempV2(d);
  for (int i = 0; i < index.ntotal; i++) {
    faiss::idx_t id = index.id_map.at(i);
    // retrieve the vector given its ID
    index.reconstruct(id, tempV2.data());
    // print ID
    std::cout << "Vector ID: " << id << "\n";
    // print vector components
    std::cout << "Components: [";
    for (int j = 0; j < d; j++) {
      std::cout << tempV2[j];
      if (j != d - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "*********************************" << std::endl;

  // Search for the top 2 most similar vectors
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