/**
 * @file postprocess.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "postprocess.hpp"
#include "logger/logger.hpp"
#include <algorithm>

namespace infer::utils {

float iou(std::array<float, 4> const &lbox, std::array<float, 4> const &rbox) {
  float interBox[] = {
      std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
      std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
      std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
      std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(BBoxes &res, std::unordered_map<int, BBoxes> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    BBoxes &dets = it->second;
    std::sort(dets.begin(), dets.end(), [](BBox const &a, BBox const &b) {
      return a.det_confidence > b.det_confidence;
    });
    for (size_t i = 0; i < dets.size(); ++i) {
      auto &item = dets[i];
      res.push_back(item);
      for (size_t j = i + 1; j < dets.size(); ++j) {
        if (iou(item.bbox, dets[j].bbox) > nms_thr) {
          dets.erase(dets.begin() + j);
          --j;
        }
      }
    }
  }
}

void nms_kbox(KeypointsBoxes &res,
              std::unordered_map<int, KeypointsBoxes> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    KeypointsBoxes &kboxes = it->second;
    std::sort(kboxes.begin(), kboxes.end(),
              [](KeypointsBox const &a, KeypointsBox const &b) {
                return a.bbox.det_confidence > b.bbox.det_confidence;
              });
    for (size_t i = 0; i < kboxes.size(); ++i) {
      auto &item = kboxes[i];
      res.push_back(item);
      for (size_t j = i + 1; j < kboxes.size(); ++j) {
        if (iou(item.bbox.bbox, kboxes[j].bbox.bbox) > nms_thr) {
          kboxes.erase(kboxes.begin() + j);
          --j;
        }
      }
    }
  }
}

void scale_ration(float &rw, float &rh, Shape const &shape,
                  Shape const &inputShape, bool isScale) {
  if (isScale) {
    rw = std::min(inputShape[0] * 1.0 / shape.at(0),
                  inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = inputShape[0] * 1.0 / shape.at(0);
    rh = inputShape[1] * 1.0 / shape.at(1);
  }
}

void restoryBox(BBox &ret, float rw, float rh, Shape const &shape) {
  int l = (ret.bbox[0] - ret.bbox[2] / 2.f) / rw;
  int t = (ret.bbox[1] - ret.bbox[3] / 2.f) / rh;
  int r = (ret.bbox[0] + ret.bbox[2] / 2.f) / rw;
  int b = (ret.bbox[1] + ret.bbox[3] / 2.f) / rh;
  ret.bbox[0] = l > 0 ? l : 0;
  ret.bbox[1] = t > 0 ? t : 0;
  ret.bbox[2] = r < shape[0] ? r : shape[0];
  ret.bbox[3] = b < shape[1] ? b : shape[1];
}

void restoryPoints(Points2f &results, float rw, float rh) {
  for (auto &point : results) {
    point.x /= rw;
    point.y /= rh;
  }
}

/**
 * @brief 恢复原始比例坐标
 *
 * @param results
 * @param shape
 * @param inputShape
 * @param isScale
 */
void restoryBoxes(BBoxes &results, Shape const &shape, Shape const &inputShape,
                  bool isScale) {
  float rw, rh;
  scale_ration(rw, rh, shape, inputShape, isScale);

  for (auto &ret : results) {
    restoryBox(ret, rw, rh, shape);
  }
}

/**
 * @brief 恢复原始比例坐标
 *
 * @param results
 * @param shape
 * @param inputShape
 * @param isScale
 */
void restoryKeypointsBoxes(KeypointsBoxes &results, Shape const &shape,
                           Shape const &inputShape, bool isScale) {
  float rw, rh;
  scale_ration(rw, rh, shape, inputShape, isScale);
  for (auto &ret : results) {
    restoryBox(ret.bbox, rw, rh, shape); // 恢复bbox
    restoryPoints(ret.points, rw, rh);   // 恢复points
  }
}

void fourPointTransform(cv::Mat &input, cv::Mat &output,
                        infer::Points2f const &points) {
  assert(points.size() == 4);
  infer::Point2f tl = points[0];
  infer::Point2f tr = points[1];
  infer::Point2f br = points[2];
  infer::Point2f bl = points[3];
  // 计算相对位置
  auto min_x_p = std::min_element(
      points.begin(), points.end(),
      [](auto const &p1, auto const &p2) { return p1.x < p2.x; });
  auto min_y_p = std::min_element(
      points.begin(), points.end(),
      [](auto const &p1, auto const &p2) { return p1.y < p2.y; });
  int min_x = min_x_p->x;
  int min_y = min_y_p->y;
  tl.x -= min_x;
  tl.y -= min_y;
  bl.x -= min_x;
  bl.y -= min_y;
  tr.x -= min_x;
  tr.y -= min_y;
  br.x -= min_x;
  br.y -= min_y;
  int widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
  int widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
  int maxWidth = std::max(widthA, widthB);

  int heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
  int heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
  int maxHeight = std::max(heightA, heightB);

  // 定义原始图像坐标和变换后的目标图像坐标
  cv::Point2f src[4] = {cv::Point2f(tl.x, tl.y), cv::Point2f(tr.x, tr.y),
                        cv::Point2f(br.x, br.y), cv::Point2f(bl.x, bl.y)};
  cv::Point2f dst[4] = {cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0),
                        cv::Point2f(maxWidth - 1, maxHeight - 1),
                        cv::Point2f(0, maxHeight - 1)};

  // 计算透视变换矩阵
  cv::Mat M = getPerspectiveTransform(src, dst);

  // 对原始图像进行透视变换
  cv::warpPerspective(input, output, M, cv::Size(maxWidth, maxHeight));
}

void sortFourPoints(Points2f &points) {
  assert(points.size() == 4);
  Point2f x1y1, x2y2, x3y3, x4y4;
  // 先对x排序，取出前两个根据y的大小决定左上和左下，后两个点根据y的大小决定右上和右下
  std::sort(points.begin(), points.end(),
            [](Point2f const &p1, Point2f const &p2) { return p1.x < p2.x; });
  if (points[0].y <= points[1].y) {
    x1y1 = points[0];
    x4y4 = points[1];
  } else {
    x1y1 = points[1];
    x4y4 = points[0];
  }
  if (points[2].y <= points[3].y) {
    x2y2 = points[2];
    x3y3 = points[3];
  } else {
    x2y2 = points[3];
    x3y3 = points[2];
  }
  points = {x1y1, x2y2, x3y3, x4y4};
  // for (auto &p : points) {
  //   std::cout << "x: " << p.x << ", "
  //             << "y: " << p.y << std::endl;
  // }
}

 size_t findClosestBBoxIndex(KeypointsBoxes const &kbboxes, float w, float h){
  float image_center_x = w / 2.0;
  float image_center_y = h / 2.0;

  float min_distance = std::numeric_limits<float>::max();
  size_t closest_bbox_index = -1;

  for (size_t i = 0; i < kbboxes.size(); ++i) {
    const auto &kbbox = kbboxes[i];
    float bbox_center_x = (kbbox.bbox.bbox[0] + kbbox.bbox.bbox[2]) / 2.0;
    float bbox_center_y = (kbbox.bbox.bbox[1] + kbbox.bbox.bbox[3]) / 2.0;

    float distance = std::hypot(bbox_center_x - image_center_x,
                                bbox_center_y - image_center_y);

    if (distance < min_distance) {
      min_distance = distance;
      closest_bbox_index = i;
    }
  }
  return closest_bbox_index;
}

} // namespace infer::utils