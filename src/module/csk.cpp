/*******************************************************************************
 * Created by zzy on 20/8.28
 * Licensed under the Simplified BSD License
 *******************************************************************************/

#include "csk.hpp"

namespace CSK {
CSKTracker::CSKTracker() {
  output_sigma_factor = 1. / 16;
  sigma = 0.2;
  lambda = 1e-2;
  // fine CSK parameters
  padding = 1.;
  interp_factor = 0.03;

  // coarse CSK parameters
  w_padding = 3; // 3 //7
  w_interp_factor = 0.03;
  downsample = 1. / 2; // 1./2 //1./3.5

  target_len = TARGET_LEN;
  target_roi_x = CSK_SEARCHING_IMG_LEN / 2 - target_len / 2;
  target_roi_y = CSK_SEARCHING_IMG_LEN / 2 - target_len / 2;
}

void CSKTracker::preprocess(cv::Mat &input, cv::Mat &output) {
  target_ratio_x = target_len / _roi_r.width;
  //    if (target_ratio_x>1.0f)
  //        target_ratio_x=sqrt(target_ratio_x);

  target_ratio_y = target_len / _roi_r.height;
  //    if (target_ratio_y<1.0f)
  //        target_ratio_y=sqrt(target_ratio_y);
  float center_x = _roi_r.x + 0.5 * _roi_r.width;
  float center_y = _roi_r.y + 0.5 * _roi_r.height;
  float left_top_x = center_x - CSK_SEARCHING_RATIO_HALF * _roi_r.width;
  float left_top_y = center_y - CSK_SEARCHING_RATIO_HALF * _roi_r.height;
  float left_top_offsetx = std::max(-left_top_x, 0.0f);
  float left_top_offsety = std::max(-left_top_y, 0.0f);

  float src_width = std::min((input.cols - left_top_x),
                             (float)(CSK_SEARCHING_RATIO * _roi_r.width)) -
                    left_top_offsetx;
  float src_height = std::min((input.rows - left_top_y),
                              (float)(CSK_SEARCHING_RATIO * _roi_r.height)) -
                     left_top_offsety;
  left_top_x = std::max(left_top_x, 0.0f);
  left_top_y = std::max(left_top_y, 0.0f);
  cv::Rect src_img_roi(left_top_x, left_top_y, src_width, src_height);
  cv::Mat _input = input(src_img_roi);
  int target_lt_offsetx = left_top_offsetx * target_ratio_x;
  int target_lt_offsety = left_top_offsety * target_ratio_y;
  int target_width = std::min((int)(_input.cols * target_ratio_x),
                              CSK_SEARCHING_IMG_LEN - target_lt_offsetx);
  int target_height = std::min((int)(_input.rows * target_ratio_y),
                               CSK_SEARCHING_IMG_LEN - target_lt_offsety);
  int target_rb_offsetx =
      CSK_SEARCHING_IMG_LEN - target_width - target_lt_offsetx;
  int target_rb_offsety =
      CSK_SEARCHING_IMG_LEN - target_height - target_lt_offsety;
  cv::resize(_input, _input, cv::Size(target_width, target_height));
  cv::copyMakeBorder(_input, output, target_lt_offsety, target_rb_offsety,
                     target_lt_offsetx, target_rb_offsetx,
                     cv::BORDER_REPLICATE);
#ifdef DISPLAY
  cv::Rect rect_coarse(48, 48, 64, 64);
  cv::Rect rect_fine(16, 16, 128, 128);
  cv::Mat _show;
  output.copyTo(_show);
  cv::rectangle(_show, rect_coarse, (255), 2);
  cv::rectangle(_show, rect_fine, (255), 2);
  cv::imshow("resized_img", _show);
  cv::waitKey(0);
#endif
}

// Initialize tracker
void CSKTracker::init(cv::Mat image, const cv::Rect &roi) {
  if (image.type() == CV_8UC3)
    cvtColor(image, image, cv::COLOR_RGB2GRAY);
  _roi_r = roi;
  preprocess(image, image);
  _roi = cv::Rect(CSK_SEARCHING_IMG_LEN / 2 - target_len / 2,
                  CSK_SEARCHING_IMG_LEN / 2 - target_len / 2, target_len,
                  target_len);
  assert(roi.width > 0 && roi.height > 0);
  _pos.x = _roi.x + _roi.width / 2;
  _pos.y = _roi.y + _roi.height / 2;

  target_sz.width = _roi.width;
  target_sz.height = _roi.height;
  float output_sigma = sqrt(float(target_sz.area())) * output_sigma_factor;
  // coarse tracker initial
  w_sz = scale_size(target_sz, (1.0 + w_padding));
  w_rsz = scale_size(w_sz, downsample);
  w_hann = createHanningMats(w_rsz);
  _w_tmpl = GetBgdWindow(image, w_sz, w_rsz);
  _w_prob = CreateGaussian2(w_rsz, output_sigma, CV_32F);
  _w_alphaf = cv::Mat(w_rsz, CV_32FC2, float(0));
  train(_w_tmpl, 1.0, 0); // train with initial frame
  // fine tracker initial
  sz = scale_size(target_sz, (1.0 + padding));
  CalTargetMeanStdDev(image, target_sz);
  _mean = t_mean;
  _dev = t_dev;
  hann = createHanningMats(w_rsz);
  _tmpl = GetSubWindow(image, sz);
  _prob = CreateGaussian2(sz, output_sigma, CV_32F);
  _alphaf = cv::Mat(sz, CV_32FC2, float(0));
  train(_tmpl, 1.0, 1); // train with initial fram
}

// Update position based on the new frame
std::tuple<std::array<float, 4>, float>
CSKTracker::update(cv::Mat input_image) {
  cv::Mat image;
  float peak_value;
  if (input_image.type() == CV_8UC3)
    cvtColor(input_image, input_image, cv::COLOR_RGB2GRAY);
  preprocess(input_image, image);
  _roi = cv::Rect(CSK_SEARCHING_IMG_LEN / 2 - target_len / 2,
                  CSK_SEARCHING_IMG_LEN / 2 - target_len / 2, target_len,
                  target_len);

  //  auto start = std::chrono::system_clock::now();
  if (_roi.x + _roi.width <= 0)
    _roi.x = -_roi.width + 1;
  if (_roi.y + _roi.height <= 0)
    _roi.y = -_roi.height + 1;
  if (_roi.x >= image.cols - 1)
    _roi.x = image.cols - 2;
  if (_roi.y >= image.rows - 1)
    _roi.y = image.rows - 2;

  // coarse CSK
  detect_coarse(_w_tmpl, GetBgdWindow(image, w_sz, w_rsz), peak_value);
  // fine CSK
  detect_fine(_tmpl, GetSubWindow(image, sz), peak_value);
  _roi.x = _pos.x - _roi.width / 2.0f;
  _roi.y = _pos.y - _roi.height / 2.0f;
  if (_roi.x >= image.cols - 1)
    _roi.x = image.cols - 1;
  if (_roi.y >= image.rows - 1)
    _roi.y = image.rows - 1;
  if (_roi.x + _roi.width <= 0)
    _roi.x = -_roi.width + 2;
  if (_roi.y + _roi.height <= 0)
    _roi.y = -_roi.height + 2;
  assert(_roi.width >= 0 && _roi.height >= 0);

  if (peak_value >= 0.3) {
    _roi_r.x += (_roi.x - target_roi_x) / target_ratio_x;
    _roi_r.y += (_roi.y - target_roi_y) / target_ratio_y;
    _roi_r.width *= _roi.width / target_len;
    _roi_r.height *= _roi.height / target_len;
    if (_roi_r.x >= input_image.cols - 1)
      _roi_r.x = input_image.cols - 1;
    if (_roi_r.y >= input_image.rows - 1)
      _roi_r.y = input_image.rows - 1;
    if (_roi_r.x + _roi_r.width <= 0)
      _roi_r.x = -_roi_r.width + 2;
    if (_roi_r.y + _roi_r.height <= 0)
      _roi_r.y = -_roi_r.height + 2;
    assert(_roi_r.width >= 0 && _roi_r.height >= 0);
    // feature for train
    cv::Mat w_x = GetBgdWindow(image, w_sz, w_rsz);
    CalTargetMeanStdDev(image, target_sz);
    _mean = (1 - interp_factor) * _mean + (interp_factor)*t_mean;
    _dev = (1 - interp_factor) * _dev + (interp_factor)*t_dev;
    cv::Mat x = GetSubWindow(image, sz);

    train(w_x, w_interp_factor, 0);
    train(x, interp_factor, 1);
  }
  return {{_roi_r.x, _roi_r.y, _roi_r.width, _roi_r.height},
          peak_value}; //_roi;
}

// Detect object in the current frame.
void CSKTracker::detect_coarse(cv::Mat tmpl_img, cv::Mat src_img,
                               float &peak_value) {
  using namespace FFTTools;
  // 1.2. Calculate the dense Gaussian kernel
  cv::Mat k;
  DenseGaussKernel(sigma, src_img, tmpl_img, k);
  cv::Mat res = (real(fftd(complexMultiplication(_w_alphaf, fftd(k)), true)));
#ifdef DISPLAY
  //    double _max,_min;
  //    minMaxLoc(res,&_min,&_max);
  //    cv::Mat _map = (res-_min)/_max*255.0f;
  cv::Mat _map = cv::max(cv::min(res, 1.0f), 0.0f) * 255.0f;
  _map.convertTo(_map, CV_8U);
  cv::resize(_map, _map, cv::Size(256, 256));
  cv::applyColorMap(_map, _map, cv::COLORMAP_JET);

  cv::imshow("feature_coarse", _map);

#endif
  // 1.3. Find the location of the max response
  cv::Point2i pi;
  double pv;
  minMaxLoc(res, NULL, &pv, NULL, &pi);
  peak_value = (float)pv;

  cv::Point2f p((float)pi.x, (float)pi.y);
  _pos.x = _pos.x + (p.x - cvFloor(float(w_rsz.width) / 2.0)) / downsample + 1;
  _pos.y = _pos.y + (p.y - cvFloor(float(w_rsz.height) / 2.0)) / downsample + 1;
}

// Detect object in the current frame.
void CSKTracker::detect_fine(cv::Mat tmpl_img, cv::Mat src_img,
                             float &peak_value) {
  using namespace FFTTools;
  // 1.2. Calculate the dense Gaussian kernel
  cv::Mat k;
  DenseGaussKernel(sigma, src_img, tmpl_img, k);
  cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
#ifdef DISPLAY
  //    double _max,_min;
  //    minMaxLoc(res,&_min,&_max);
  //    cv::Mat _map = (res-_min)/_max*255.0f;
  cv::Mat _map = cv::max(cv::min(res, 1.0f), 0.0f) * 255.0f;
  _map.convertTo(_map, CV_8U);
  cv::resize(_map, _map, cv::Size(256, 256));
  cv::applyColorMap(_map, _map, cv::COLORMAP_JET);

  cv::imshow("feature_fine", _map);

#endif
  // 1.3. Find the location of the max response
  cv::Point2i pi;
  double pv;
  minMaxLoc(res, NULL, &pv, NULL, &pi);
  peak_value = (float)pv;

  cv::Point2f p((float)pi.x, (float)pi.y);
  _pos.x = _pos.x + p.x - cvFloor(float(sz.width) / 2.0) + 1;
  _pos.y = _pos.y + p.y - cvFloor(float(sz.height) / 2.0) + 1;
}

// train tracker with a single image
void CSKTracker::train(cv::Mat x, float train_interp_factor, bool fine) {
  using namespace FFTTools;
  cv::Mat k;
  DenseGaussKernel(sigma, x, x, k);
  if (fine) {
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x;
    _alphaf =
        (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf;
  } else {
    cv::Mat alphaf = complexDivision(_w_prob, (fftd(k) + lambda));
    _w_tmpl = (1 - train_interp_factor) * _w_tmpl + (train_interp_factor)*x;
    _w_alphaf =
        (1 - train_interp_factor) * _w_alphaf + (train_interp_factor)*alphaf;
  }
}

cv::Mat CSKTracker::GetBgdWindow(const cv::Mat &frame, cv::Size sz,
                                 cv::Size rsz) {
  cv::Mat bgdWindow;
  cv::Mat subWindow;
  cv::Point lefttop(cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + 1,
                    cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + 1);
  cv::Point rightbottom(
      cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + sz.width + 1,
      cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + sz.height + 1);
  cv::Rect idea_rect(lefttop, rightbottom);
  cv::Rect true_rect = idea_rect & cv::Rect(0, 0, frame.cols, frame.rows);
  cv::Rect border(0, 0, 0, 0);
  if (true_rect.area() == 0) {
    int x_start, x_width, y_start, y_height;

    x_start = cv::min(frame.cols - 1, cv::max(0, idea_rect.x));
    x_width = cv::max(1, cv::min(idea_rect.x + idea_rect.width, frame.cols) -
                             x_start);
    y_start = cv::min(frame.rows - 1, cv::max(0, idea_rect.y));
    y_height = cv::max(1, cv::min(idea_rect.y + idea_rect.height, frame.rows) -
                              y_start);

    true_rect = cv::Rect(x_start, y_start, x_width, y_height);

    if ((idea_rect.x + idea_rect.width - 1) < 0)
      border.x = sz.width - 1;
    else if (idea_rect.x > (frame.cols - 1))
      border.width = sz.width - 1;
    else {
      if (idea_rect.x < 0)
        border.x = -idea_rect.x;
      if ((idea_rect.x + idea_rect.width) > frame.cols)
        border.width = idea_rect.x + idea_rect.width - frame.cols;
    }

    if ((idea_rect.y + idea_rect.height - 1) < 0)
      border.y = sz.height - 1;
    else if (idea_rect.y > (frame.rows - 1))
      border.height = sz.height - 1;
    else {
      if (idea_rect.y < 0)
        border.y = -idea_rect.y;
      if ((idea_rect.y + idea_rect.height) > frame.rows)
        border.height = idea_rect.y + idea_rect.height - frame.rows;
    }

    frame(true_rect).copyTo(bgdWindow);

  } else if (true_rect.area() == idea_rect.area()) {
    frame(true_rect).copyTo(bgdWindow);
  } else {
    frame(true_rect).copyTo(bgdWindow);
    border.y = true_rect.y - idea_rect.y;
    border.height =
        idea_rect.y + idea_rect.height - true_rect.y - true_rect.height;
    border.x = true_rect.x - idea_rect.x;
    border.width =
        idea_rect.x + idea_rect.width - true_rect.x - true_rect.width;
  }

  if (border != cv::Rect(0, 0, 0, 0)) {
    cv::copyMakeBorder(bgdWindow, bgdWindow, border.y, border.height, border.x,
                       border.width, cv::BORDER_REPLICATE);
  }

  cv::resize(bgdWindow, subWindow, rsz, 0, 0, cv::INTER_NEAREST);
  cv::Scalar mean_sub;
  cv::Scalar stdv_sub;
  cv::meanStdDev(subWindow, mean_sub, stdv_sub);
  float mean = mean_sub[0];
  float dev = stdv_sub[0];
  float stretch = ((dev)*2);
  subWindow.convertTo(subWindow, CV_32FC1, 1.0 / stretch, -mean / stretch);
  // subWindow.convertTo(subWindow, CV_32FC1,1.0/255.0,-0.5);

  subWindow = subWindow.mul(w_hann);
  return subWindow;
}

void CSKTracker::CalTargetMeanStdDev(const cv::Mat &frame, cv::Size sz) {
  cv::Mat subWindow;
  cv::Point lefttop(cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + 1,
                    cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + 1);
  cv::Point rightbottom(
      cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + sz.width + 1,
      cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + sz.height + 1);
  cv::Rect idea_rect = _roi;
  cv::Rect true_rect = idea_rect & cv::Rect(0, 0, frame.cols, frame.rows);
  cv::Rect border(0, 0, 0, 0);
  if (true_rect.area() == 0) {
    int x_start, x_width, y_start, y_height;

    x_start = cv::min(frame.cols - 1, cv::max(0, idea_rect.x));
    x_width = cv::max(1, cv::min(idea_rect.x + idea_rect.width, frame.cols) -
                             x_start);
    y_start = cv::min(frame.rows - 1, cv::max(0, idea_rect.y));
    y_height = cv::max(1, cv::min(idea_rect.y + idea_rect.height, frame.rows) -
                              y_start);

    true_rect = cv::Rect(x_start, y_start, x_width, y_height);

    if ((idea_rect.x + idea_rect.width - 1) < 0)
      border.x = sz.width - 1;
    else if (idea_rect.x > (frame.cols - 1))
      border.width = sz.width - 1;
    else {
      if (idea_rect.x < 0)
        border.x = -idea_rect.x;
      if ((idea_rect.x + idea_rect.width) > frame.cols)
        border.width = idea_rect.x + idea_rect.width - frame.cols;
    }

    if ((idea_rect.y + idea_rect.height - 1) < 0)
      border.y = sz.height - 1;
    else if (idea_rect.y > (frame.rows - 1))
      border.height = sz.height - 1;
    else {
      if (idea_rect.y < 0)
        border.y = -idea_rect.y;
      if ((idea_rect.y + idea_rect.height) > frame.rows)
        border.height = idea_rect.y + idea_rect.height - frame.rows;
    }

    frame(true_rect).copyTo(subWindow);
  } else if (true_rect.area() == idea_rect.area()) {
    frame(true_rect).copyTo(subWindow);
  } else {
    frame(true_rect).copyTo(subWindow);
    border.y = true_rect.y - idea_rect.y;
    border.height =
        idea_rect.y + idea_rect.height - true_rect.y - true_rect.height;
    border.x = true_rect.x - idea_rect.x;
    border.width =
        idea_rect.x + idea_rect.width - true_rect.x - true_rect.width;
  }

  if (border != cv::Rect(0, 0, 0, 0)) {
    cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x,
                       border.width, cv::BORDER_REPLICATE);
  }
  cv::Scalar mean_sub;
  cv::Scalar stdv_sub;
  cv::meanStdDev(subWindow, mean_sub, stdv_sub);
  t_mean = mean_sub[0];
  t_dev = stdv_sub[0];
  // printf("mean_sub: %.3f stdv_sub: %.3f\n", *mean,*dev);
}

cv::Mat CSKTracker::GetSubWindow(const cv::Mat &frame, cv::Size sz) {
  cv::Mat subWindow;
  cv::Point lefttop(cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + 1,
                    cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + 1);
  cv::Point rightbottom(
      cvFloor(_pos.x) - cvFloor(float(sz.width) / 2.0) + sz.width + 1,
      cvFloor(_pos.y) - cvFloor(float(sz.height) / 2.0) + sz.height + 1);
  cv::Rect idea_rect(lefttop, rightbottom);
  cv::Rect true_rect = idea_rect & cv::Rect(0, 0, frame.cols, frame.rows);
  cv::Rect border(0, 0, 0, 0);
  if (true_rect.area() == 0) {
    int x_start, x_width, y_start, y_height;

    x_start = cv::min(frame.cols - 1, cv::max(0, idea_rect.x));
    x_width = cv::max(1, cv::min(idea_rect.x + idea_rect.width, frame.cols) -
                             x_start);
    y_start = cv::min(frame.rows - 1, cv::max(0, idea_rect.y));
    y_height = cv::max(1, cv::min(idea_rect.y + idea_rect.height, frame.rows) -
                              y_start);

    true_rect = cv::Rect(x_start, y_start, x_width, y_height);

    if ((idea_rect.x + idea_rect.width - 1) < 0)
      border.x = sz.width - 1;
    else if (idea_rect.x > (frame.cols - 1))
      border.width = sz.width - 1;
    else {
      if (idea_rect.x < 0)
        border.x = -idea_rect.x;
      if ((idea_rect.x + idea_rect.width) > frame.cols)
        border.width = idea_rect.x + idea_rect.width - frame.cols;
    }

    if ((idea_rect.y + idea_rect.height - 1) < 0)
      border.y = sz.height - 1;
    else if (idea_rect.y > (frame.rows - 1))
      border.height = sz.height - 1;
    else {
      if (idea_rect.y < 0)
        border.y = -idea_rect.y;
      if ((idea_rect.y + idea_rect.height) > frame.rows)
        border.height = idea_rect.y + idea_rect.height - frame.rows;
    }

    frame(true_rect).copyTo(subWindow);
  } else if (true_rect.area() == idea_rect.area()) {
    frame(true_rect).copyTo(subWindow);
  } else {
    frame(true_rect).copyTo(subWindow);
    border.y = true_rect.y - idea_rect.y;
    border.height =
        idea_rect.y + idea_rect.height - true_rect.y - true_rect.height;
    border.x = true_rect.x - idea_rect.x;
    border.width =
        idea_rect.x + idea_rect.width - true_rect.x - true_rect.width;
  }

  if (border != cv::Rect(0, 0, 0, 0)) {
    cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x,
                       border.width, cv::BORDER_REPLICATE);
  }

  float stretch = ((_dev)*2);
  subWindow.convertTo(subWindow, CV_32FC1, 1.0 / stretch, -_mean / stretch);
  // subWindow.convertTo(subWindow, CV_32FC1,1.0/255.0,-0.5);

  subWindow = subWindow.mul(hann);
  return subWindow;
}

// Initialize Hanning window. Function called only in the first frame.
cv::Mat CSKTracker::createHanningMats(cv::Size sz) {
  cv::Mat hann1t = cv::Mat(cv::Size(sz.width, 1), CV_32F, cv::Scalar(0));
  cv::Mat hann2t = cv::Mat(cv::Size(1, sz.height), CV_32F, cv::Scalar(0));
  for (int i = 0; i < hann1t.cols; i++) {
    hann1t.at<float>(0, i) =
        0.5 *
        (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
  }
  for (int i = 0; i < hann2t.rows; i++) {
    hann2t.at<float>(i, 0) =
        0.5 *
        (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
  }
  cv::Mat hann2d = hann2t * hann1t;
  return hann2d;
}

void CSKTracker::DenseGaussKernel(float sigma, const cv::Mat &x,
                                  const cv::Mat &y, cv::Mat &k) {
  cv::Mat xf, yf;
  dft(x, xf, cv::DFT_COMPLEX_OUTPUT);
  dft(y, yf, cv::DFT_COMPLEX_OUTPUT);
  float xx = norm(x);
  xx = xx * xx;
  float yy = norm(y);
  yy = yy * yy;

  cv::Mat xyf;
  mulSpectrums(xf, yf, xyf, 0, true);

  cv::Mat xy;
  idft(xyf, xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
  CircShift(xy, scale_size(x.size(), 0.5));
  float numelx1 = x.cols * x.rows;
  exp((-1 / (sigma * sigma)) * max(0, (xx + yy - 2 * xy) / numelx1), k);
}

cv::Mat CSKTracker::CreateGaussian1(int n, float sigma, int ktype) {
  CV_Assert(ktype == CV_32F || ktype == CV_32F);
  cv::Mat kernel(n, 1, ktype);
  float *cf = kernel.ptr<float>();
  float *cd = kernel.ptr<float>();

  float sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
  float scale2X = -0.5 / (sigmaX * sigmaX);

  int i;
  for (i = 0; i < n; i++) {
    float x = i - floor((float)(n / 2)) + 1;
    float t = std::exp(scale2X * x * x);
    if (ktype == CV_32F) {
      cf[i] = (float)t;
    } else {
      cd[i] = t;
    }
  }
  return kernel;
}

cv::Mat CSKTracker::CreateGaussian2(cv::Size sz, float sigma, int ktype) {
  cv::Mat a = CreateGaussian1(sz.height, sigma, ktype);
  cv::Mat b = CreateGaussian1(sz.width, sigma, ktype);
  cv::Mat result;
  dft(a * b.t(), result, cv::DFT_COMPLEX_OUTPUT);
  return result;
}

void CSKTracker::CircShift(cv::Mat &x, cv::Size sz) {
  int cx, cy;
  if (sz.width < 0)
    cx = -sz.width;
  else
    cx = x.cols - sz.width;

  if (sz.height < 0)
    cy = -sz.height;
  else
    cy = x.rows - sz.height;

  cv::Mat q0(x, cv::Rect(0, 0, cx,
                         cy)); // Top-Left - Create a ROI per quadrant
  cv::Mat q1(x, cv::Rect(cx, 0, x.cols - cx, cy));           // Top-Right
  cv::Mat q2(x, cv::Rect(0, cy, cx, x.rows - cy));           // Bottom-Left
  cv::Mat q3(x, cv::Rect(cx, cy, x.cols - cx, x.rows - cy)); // Bottom-Right

  cv::Mat tmp1, tmp2; // swap quadrants (Top-Left with Bottom-Right)
  hconcat(q3, q2, tmp1);
  hconcat(q1, q0, tmp2);
  vconcat(tmp1, tmp2, x);
}
} // namespace CSK