/**
 * @file facelib.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "logger/logger.hpp"
#include <cmath>
#include <cstdlib>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <iostream>
#include <memory>
#include <vector>

#ifndef __INFER_FACE_LIBRARY_H_
#define __INFER_FACE_LIBRARY_H_

namespace infer::solution {
class FaceLibrary {
private:
  int d; // dimensionality of the vectors
  std::unique_ptr<faiss::IndexFlatIP> flatIndex;
  std::unique_ptr<faiss::IndexIDMap2> index;

public:
  FaceLibrary(int dim) : d(dim) {
    flatIndex = std::make_unique<faiss::IndexFlatIP>(dim);
    index = std::make_unique<faiss::IndexIDMap2>(flatIndex.get());
  }

  void addVector(float *vec, long id) {
    // normalize_L2(vec);
    index->add_with_ids(1, vec, &id);
  }

  void addVectors(float *vecs, std::vector<long> ids) {
    // normalize_L2(vec);
    index->add_with_ids(ids.size(), vecs, ids.data());
  }

  void deleteVector(long id) {
    faiss::IDSelectorArray selector{1, &id};
    index->remove_ids(selector);
  }

  void deleteVectors(std::vector<long> &ids) {
    faiss::IDSelectorArray selector{ids.size(), ids.data()};
    index->remove_ids(selector);
  }

  void updateVector(faiss::idx_t id, float *new_vec) {
    deleteVector(id);
    addVector(new_vec, id);
  }

  std::vector<std::pair<faiss::idx_t, float>> search(float *query_vec, int k) {
    // normalize_L2(query_vec);
    float *distances = new float[k];
    int64_t *indices = new int64_t[k];

    index->search(1, query_vec, k, distances, indices);

    std::vector<std::pair<faiss::idx_t, float>> results;
    for (int i = 0; i < k; i++) {
      results.push_back({indices[i], distances[i]});
    }

    delete[] distances;
    delete[] indices;

    return results;
  }

  // Save the index to a file
  void saveToFile(std::string const &filename) {
    // Using Faiss's write_index function
    faiss::write_index(index.get(), filename.c_str());
  }

  // Load the index from a file
  bool loadFromFile(std::string const &filename) {
    // Release the current index memory
    index.reset(nullptr);

    // Using Faiss's read_index function
    faiss::IndexIDMap2 *loadedIndex = dynamic_cast<faiss::IndexIDMap2 *>(
        faiss::read_index(filename.c_str(), faiss::IO_FLAG_MMAP));
    if (loadedIndex) {
      index.reset(loadedIndex);
    } else {
      FLOWENGINE_LOGGER_ERROR("FaceLibrary: failed to load the facelib file{}",
                              filename);
    }
    return true;
  }

  void printVectors() {
    std::vector<float> vec(d); // temporary buffer for storing vector components

    for (int i = 0; i < index->ntotal; i++) {
      faiss::idx_t id = index->id_map.at(i);

      // retrieve the vector given its ID
      index->reconstruct(id, vec.data());

      // print ID
      std::cout << "Vector ID: " << id << "\n";

      // print vector components
      std::cout << "Components: [";
      for (int j = 0; j < d; j++) {
        std::cout << vec[j];
        if (j != d - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
    std::cout << "******************************************" << std::endl;
  }
};
} // namespace infer::solution

#endif