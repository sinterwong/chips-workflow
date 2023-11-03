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
#include "preprocess.hpp"
#include <cmath>
#include <cstdlib>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#ifndef __INFER_FACE_LIBRARY_H_
#define __INFER_FACE_LIBRARY_H_

namespace server::face::core {
class FaceLibrary {
private:
  int d; // dimensionality of the vectors
  std::unique_ptr<faiss::IndexFlatIP> flatIndex;
  std::unique_ptr<faiss::IndexIDMap2> index;
  std::unordered_set<faiss::idx_t> existing_ids; // 跟踪已添加的向量
  std::shared_mutex m;

public:
  FaceLibrary(int dim) : d(dim) {
    flatIndex = std::make_unique<faiss::IndexFlatIP>(dim);
    index = std::make_unique<faiss::IndexIDMap2>(flatIndex.get());
  }

  bool addVector(float *vec, long id) {
    std::lock_guard lk(m);
    if (existing_ids.find(id) == existing_ids.end()) {
      // id 不存在，执行新增操作
      index->add_with_ids(1, vec, &id);
      existing_ids.insert(id);
    } else {
      FLOWENGINE_LOGGER_ERROR("ID {} reuse.", id);
      return false;
    }
    return true;
  }

  // void addVectors(float *vecs, std::vector<long> ids) {
  //   std::lock_guard lk(m);
  //   index->add_with_ids(ids.size(), vecs, ids.data());
  // }

  bool deleteVector(long id) {
    faiss::IDSelectorArray selector{1, &id};
    std::lock_guard lk(m);
    if (existing_ids.find(id) != existing_ids.end()) {
      // id 存在，执行后面操作即可
      index->remove_ids(selector);
      existing_ids.erase(id);
    } else {
      FLOWENGINE_LOGGER_ERROR("ID {} does not exist.", id);
      return false;
    }
    return true;
  }

  // void deleteVectors(std::vector<long> &ids) {
  //   faiss::IDSelectorArray selector{ids.size(), ids.data()};
  //   std::lock_guard lk(m);
  //   index->remove_ids(selector);
  // }

  bool updateVector(faiss::idx_t id, float *new_vec) {
    faiss::IDSelectorArray selector{1, &id};
    std::lock_guard lk(m);
    if (existing_ids.find(id) != existing_ids.end()) {
      // id 存在，执行后面操作即可
      index->remove_ids(selector);
      index->add_with_ids(1, new_vec, &id);
    } else {
      FLOWENGINE_LOGGER_ERROR("ID {} does not exist.", id);
      return false;
    }
    return true;
  }

  std::vector<std::pair<faiss::idx_t, float>> search(float *query_vec, int k) {
    float *distances = new float[k];
    int64_t *indices = new int64_t[k];

    {
      std::shared_lock lk(m);
      index->search(1, query_vec, k, distances, indices);
    }

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
    std::lock_guard lk(m);
    // Using Faiss's write_index function
    faiss::write_index(index.get(), filename.c_str());
  }

  // Load the index from a file
  bool loadFromFile(std::string const &filename) {
    std::lock_guard lk(m);
    // Release the current index memory
    index.reset(nullptr);

    // Using Faiss's read_index function
    faiss::IndexIDMap2 *loadedIndex = dynamic_cast<faiss::IndexIDMap2 *>(
        faiss::read_index(filename.c_str(), faiss::IO_FLAG_MMAP));
    if (loadedIndex) {
      index.reset(loadedIndex);
      // 直接访问 id_map 字段以获取所有的 ID
      const auto &id_map = index->id_map;
      // 遍历 id_map，恢复索引表
      for (size_t i = 0; i < id_map.size(); ++i) {
        existing_ids.insert(id_map[i]);
      }
    } else {
      FLOWENGINE_LOGGER_ERROR("FaceLibrary: failed to load the facelib file{}",
                              filename);
    }
    return true;
  }

  void printVectors() {
    std::vector<float> vec(d); // temporary buffer for storing vector components

    std::shared_lock lk(m);
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
} // namespace server::face::core

#endif