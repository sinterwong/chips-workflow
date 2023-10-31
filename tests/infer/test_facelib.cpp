/**
 * @file test_facelib.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "facelib.hpp"
#include "preprocess.hpp"
#include <gflags/gflags.h>
#include <numeric>
#include <vector>

DEFINE_string(img, "", "Specify a face image path.");
DEFINE_string(model_path, "", "Specify the yolo model path.");

using namespace infer;
using namespace solution;

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize random seed
  srand(666);

  // Number of vectors and vector dimensionality
  int nb = 10;
  int d = 8;

  // Randomly generate vecotors and normalize them
  float *xb = new float[d * nb];
  for (int i = 0; i < nb * d; i++) {
    xb[i] = rand() % 1000;
  }
  for (int i = 0; i < nb; i++) {
    utils::normalize_L2(xb + i * d, d);
  }

  // Generate a random query vector and normalize it
  float *xq = new float[d];
  for (int i = 0; i < d; i++) {
    xq[i] = rand() % 1000;
  }
  utils::normalize_L2(xq, d);

  solution::FaceLibrary facelib{d};

  std::vector<long> numbers(nb);
  std::iota(numbers.begin(), numbers.end(), 0);
  facelib.addVectors(xb, numbers);

  facelib.printVectors();

  std::vector<long> remove_ids = {1, 3, 8};
  facelib.deleteVectors(remove_ids);

  facelib.printVectors();

  facelib.updateVector(4, xq);

  auto results = facelib.search(xq, 2);
  for (auto &ret : results) {
    std::cout << "Index: " << ret.first << ", "
              << "Similarity: " << ret.second << std::endl;
  }

  facelib.saveToFile("output.bin");

  solution::FaceLibrary facelib2{d};
  facelib2.loadFromFile("output.bin");
  facelib2.printVectors();

  gflags::ShutDownCommandLineFlags();
  return 0;
}