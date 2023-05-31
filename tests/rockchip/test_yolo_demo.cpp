#include "infer/postprocess.hpp"
#include "logger/logger.hpp"
#include "rga.h"
#include "rknn_api.h"
#include "utils/time_utils.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <RgaUtils.h>
#include <gflags/gflags.h>
#include <im2d.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <vector>

DEFINE_string(model_path, "", "Specify the path of dnn model.");
DEFINE_string(image_path, "", "Specify the path of image.");

std::vector<unsigned char> load_data(std::ifstream &ifs, std::size_t ofst,
                                     std::size_t sz) {
  std::vector<unsigned char> data(sz);
  if (!ifs) {
    std::cerr << "Error: file stream is not open." << std::endl;
    return {};
  }

  if (!ifs.seekg(ofst)) {
    std::cerr << "Error: failed to seek in the file." << std::endl;
    return {};
  }

  if (!ifs.read(reinterpret_cast<char *>(data.data()), sz)) {
    std::cerr << "Error: failed to read data from the file." << std::endl;
    return {};
  }

  return data;
}

std::vector<unsigned char> load_model(const std::string &filename,
                                      int &model_size) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "Error: failed to open file " << filename << std::endl;
    return {};
  }

  ifs.seekg(0, std::ios::end);
  std::size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<unsigned char> data = load_data(ifs, 0, size);
  ifs.close();

  model_size = size;
  return data;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

// static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
//   unsigned char *data;
//   int ret;

//   data = NULL;

//   if (NULL == fp) {
//     return NULL;
//   }

//   ret = fseek(fp, ofst, SEEK_SET);
//   if (ret != 0) {
//     printf("blob seek failure.\n");
//     return NULL;
//   }

//   data = (unsigned char *)malloc(sz);
//   if (data == NULL) {
//     printf("buffer malloc failure.\n");
//     return NULL;
//   }
//   ret = fread(data, 1, sz, fp);
//   return data;
// }

// static unsigned char *load_model(const char *filename, int *model_size) {
//   FILE *fp;
//   unsigned char *data;

//   fp = fopen(filename, "rb");
//   if (NULL == fp) {
//     printf("Open file %s failed.\n", filename);
//     return NULL;
//   }

//   fseek(fp, 0, SEEK_END);
//   int size = ftell(fp);

//   data = load_data(fp, 0, size);

//   fclose(fp);

//   *model_size = size;
//   return data;
// }

static void dump_tensor_attr(rknn_tensor_attr *attr) {
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
         "size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
         attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
         get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

void generateBoxes(std::unordered_map<int, common::BBoxes> &m, void **outputs,
                   int numAnchors, int n) {
  float **output = reinterpret_cast<float **>(*outputs);
  for (int j = 0; j < numAnchors * n; j += n) {
    if (output[0][j + 4] <= 0.3)
      continue;
    common::BBox det;
    det.class_id =
        std::distance(output[0] + j + 5,
                      std::max_element(output[0] + j + 5, output[0] + j + n));
    int real_idx = j + 5 + det.class_id;
    det.det_confidence = output[0][real_idx];
    memcpy(&det, &output[0][j], 5 * sizeof(float));
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, common::BBoxes());
    m[det.class_id].push_back(det);
  }
}

int main(int argc, char **argv) {
  // FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  rknn_context ctx;

  // init rga context
  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  int ret;

  cv::Mat orig_img = cv::imread(FLAGS_image_path, 1);
  if (!orig_img.data) {
    FLOWENGINE_LOGGER_ERROR("read image failed!");
    return -1;
  }

  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

  int img_width = img.cols, img_height = img.rows;
  FLOWENGINE_LOGGER_INFO("image size: {}x{}", img_width, img_height);

  // loading model
  int model_data_size = 0;
  // unsigned char *model_data = load_model(FLAGS_model_path.c_str(),
  // &model_data_size); ret = rknn_init(&ctx, model_data, model_data_size, 0,
  // nullptr);

  std::vector<unsigned char> model_data =
      load_model(FLAGS_model_path, model_data_size);
  ret = rknn_init(&ctx, model_data.data(), model_data_size, 0, nullptr);

  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("rknn_init failed!");
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                   sizeof(rknn_sdk_version));
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("rknn_query version failed!");
    return -1;
  }

  FLOWENGINE_LOGGER_INFO("sdk version: {} driver version: {}",
                         version.api_version, version.drv_version);

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("rknn_query input output num failed!");
    return -1;
  }

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (unsigned int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret < 0) {
      FLOWENGINE_LOGGER_ERROR("rknn_query input attr failed!");
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (unsigned int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
    width = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    height = input_attrs[0].dims[1];
    width = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }
  FLOWENGINE_LOGGER_INFO("model input height={}, width={}, channel={}", height,
                         width, channel);
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = width * height * channel;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  void *resize_buf = nullptr;
  if (img_width != width || img_height != height) {
    FLOWENGINE_LOGGER_INFO("resize with RGA");
    resize_buf = malloc(height * width * channel);
    memset(resize_buf, 0x00, height * width * channel);

    src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height,
                                 RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, width, height,
                                 RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
      FLOWENGINE_LOGGER_ERROR("imcheck failed");
      return -1;
    }
    // IM_STATUS STATUS = imresize(src, dst);

    // for debug
    cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);
    cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;
  } else {
    inputs[0].buf = (void *)img.data;
  }

  rknn_inputs_set(ctx, io_num.n_input, inputs);

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (unsigned int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = 0;
  }

  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

  std::vector<float> out_scales;
  std::vector<int32_t> out_zps;
  for (unsigned int i = 0; i < io_num.n_output; ++i) {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }

  std::unordered_map<int, common::BBoxes> m;
  int numAnchors = output_attrs[0].dims[1], n = output_attrs[0].dims[2];

  float thres = 0.25;
  thres = unsigmoid(thres);
  int8_t thres_i8 = qnt_f32_to_affine(thres, out_zps[0], out_scales[0]);

  // for debug
  int8_t *out_buf = reinterpret_cast<int8_t *>(outputs[0].buf);
  std::vector<float> out;
  for (int i = 0; i < numAnchors * n; i++) {
    if ((i + 1) % 6 == 0 && out_buf[i - 1] > thres_i8) {
      std::cout << out_buf[i - 1] << ", ";
    };
    out.push_back(deqnt_affine_to_f32(out_buf[i], out_zps[0], out_scales[0]));
  }
  std::cout << std::endl;
  void *outs[1];
  outs[0] = reinterpret_cast<void *>(out.data());
  void *output = outs;
  generateBoxes(m, &output, numAnchors, n);

  common::BBoxes bboxes;
  infer::utils::nms(bboxes, m, 0.45);
  infer::utils::restoryBoxes(bboxes, {img_width, img_height, 3},
                             {width, height, 3}, false);

  FLOWENGINE_LOGGER_INFO("number of result: {}", bboxes.size());
  for (auto &bbox : bboxes) {
    // cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
    //               bbox.bbox[3] - bbox.bbox[1]);
    // cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);

    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("test_rockchip_yolo_demo_out.jpg", img);

  // release
  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  ret = rknn_destroy(ctx);

  // if (model_data) {
  //   free(model_data);
  // }
  if (resize_buf) {
    free(resize_buf);
  }

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}
