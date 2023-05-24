#include "ffstream.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"

#include <cstring>
#include <vector>

#include <gflags/gflags.h>
#include <osal/mpp_common.h>
#include <osal/mpp_mem.h>
#include <rk_mpi.h>

DEFINE_string(uri, "", "Specify the uri of video.");

#define BH_RTSP_READER_BUF_LEN 2000000 // 2MB

struct DataCrc {
  RK_U32 len;
  RK_U32 sum_cnt;
  RK_ULONG *sum;
  RK_U32 vor; // value of the xor
};

struct FrmCrc {
  DataCrc luma;
  DataCrc chroma;
};

struct MpiDecLoopData {
  MppCtx ctx;
  MppApi *mpi;
  RK_U32 quiet;

  /* end of stream flag when set quit the loop */
  RK_U32 loop_end;

  /* input and output */
  MppBufferGroup frm_grp;
  MppPacket packet;
  MppFrame frame;

  FILE *fp_output;
  RK_S32 frame_count;
  RK_S32 frame_num;

  RK_S64 first_pkt;
  RK_S64 first_frm;

  size_t max_usage;
  float frame_rate;
  RK_S64 elapsed_time;
  RK_S64 delay;
  FILE *fp_verify;
  FrmCrc checkcrc;
};

RK_S64 mpp_time() {
  struct timespec time = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (RK_S64)time.tv_sec * 1000000 + (RK_S64)time.tv_nsec / 1000;
}

void *thread_decode(void *arg) {
  MpiDecLoopData *data = (MpiDecLoopData *)arg;
  MppCtx ctx = data->ctx;
  MppApi *mpi = data->mpi;
  RK_S64 t_s, t_e;

  memset(&data->checkcrc, 0, sizeof(data->checkcrc));
  data->checkcrc.luma.sum = mpp_malloc(RK_ULONG, 512);
  data->checkcrc.chroma.sum = mpp_malloc(RK_ULONG, 512);

  t_s = mpp_time();

  if (1) {
    while (!data->loop_end) {
      // dec_simple(data);
    }
  }

  t_e = mpp_time();
  data->elapsed_time = t_e - t_s;
  data->frame_count = data->frame_count;
  data->frame_rate = (float)data->frame_count * 1000000 / data->elapsed_time;
  data->delay = data->first_frm - data->first_pkt;

  MPP_FREE(data->checkcrc.luma.sum);
  MPP_FREE(data->checkcrc.chroma.sum);

  return NULL;
}

int dec_decode() {

  // base flow context
  MppCtx ctx = nullptr;
  MppApi *mpi = nullptr;

  // input / output
  MppPacket packet = nullptr;
  MppFrame frame = nullptr;

  // config for runtime mode
  MppDecCfg cfg = nullptr;
  int need_split = 1; // 是否需要分割码流（取决于每次输入是否以帧为单位）

  // resources
  MppBuffer frm_buf = nullptr;
  MpiDecLoopData data;
  MPP_RET ret = MPP_OK;

  RK_U32 width = 1920;
  RK_U32 height = 1080;
  MppCodingType type = MPP_VIDEO_CodingAVC;

  int simple = (type != MPP_VIDEO_CodingMJPEG) ? (1) : (0);

  if (simple) {
    ret = mpp_packet_init(&packet, nullptr, 0);
    if (ret) {
      FLOWENGINE_LOGGER_ERROR("mpp_packet_init failed");
      goto MPP_TEST_OUT;
    }
  } else {
    RK_U32 hor_stride = MPP_ALIGN(width, 16);
    RK_U32 ver_stride = MPP_ALIGN(height, 16);

    ret = mpp_buffer_group_get_internal(&data.frm_grp, MPP_BUFFER_TYPE_ION);
    if (ret) {
      FLOWENGINE_LOGGER_ERROR("get mpp buffer group failed");
      goto MPP_TEST_OUT;
    }

    ret = mpp_frame_init(&frame);
    if (ret) {
      FLOWENGINE_LOGGER_ERROR("mpp_frame_init failed");
      goto MPP_TEST_OUT;
    }

    ret = mpp_buffer_get(data.frm_grp, &frm_buf, hor_stride * ver_stride * 4);
    if (ret) {
      FLOWENGINE_LOGGER_ERROR("mpp_buffer_get failed");
      goto MPP_TEST_OUT;
    }
    mpp_frame_set_buffer(frame, frm_buf);
  }

  // open input / output files
  data.fp_output = fopen("./output.yuv", "w+b");
  if (NULL == data.fp_output) {
    FLOWENGINE_LOGGER_ERROR("failed to open output file %s", "./output.yuv");
    goto MPP_TEST_OUT;
  }

  ret = mpp_create(&ctx, &mpi);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpp_create failed");
    goto MPP_TEST_OUT;
  }
  ret = mpp_init(ctx, MPP_CTX_DEC, type);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpp_init failed");
    goto MPP_TEST_OUT;
  }
  mpp_dec_cfg_init(&cfg);

  // get default config from decoder context
  ret = mpi->control(ctx, MPP_DEC_GET_CFG, cfg);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpi control failed");
    goto MPP_TEST_OUT;
  }

  // split_parse is to enable mpp internal frame spliter when the input packet
  // is not aplited into frames.
  ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", need_split);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpp_dec_cfg_set_u32 failed");
    goto MPP_TEST_OUT;
  }

  ret = mpi->control(ctx, MPP_DEC_SET_CFG, cfg);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpi control failed");
    goto MPP_TEST_OUT;
  }

  data.ctx = ctx;
  data.mpi = mpi;
  data.loop_end = 0;
  data.packet = packet;
  data.frame = frame;
  data.frame_count = 0;
  data.frame_num = 10;
  data.quiet = 0;

MPP_TEST_OUT:
  if (data.packet) {
    mpp_packet_deinit(&data.packet);
    data.packet = NULL;
  }

  if (frame) {
    mpp_frame_deinit(&frame);
    frame = NULL;
  }

  if (ctx) {
    mpp_destroy(ctx);
    ctx = NULL;
  }

  if (simple) {
    if (frm_buf) {
      mpp_buffer_put(frm_buf);
      frm_buf = NULL;
    }
  }

  if (data.frm_grp) {
    mpp_buffer_group_put(data.frm_grp);
    data.frm_grp = NULL;
  }

  if (data.fp_output) {
    fclose(data.fp_output);
    data.fp_output = NULL;
  }

  if (data.fp_verify) {
    fclose(data.fp_verify);
    data.fp_verify = NULL;
  }

  if (cfg) {
    mpp_dec_cfg_deinit(cfg);
    cfg = NULL;
  }

  // pthread_attr_destroy(&attr);

  return ret;
}

int main(int argc, char **argv) {
  // FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /*
    // open stream
    video::utils::FFStream stream(FLAGS_uri);
    if (!stream.openStream()) {
      FLOWENGINE_LOGGER_ERROR("can't open the stream {}!",
                              std::string(FLAGS_uri));
      return -1;
    }
    FLOWENGINE_LOGGER_INFO("video is opened!");
    FLOWENGINE_LOGGER_INFO("width: {}, height: {}, rate: {}", stream.getWidth(),
                           stream.getHeight(), stream.getRate());

    // 指向了ffmpeg读取的裸流数据(unsafe)
    void *raw_data; // 外部导入类型buffer
    while (stream.isRunning()) {
      stream.getRawFrame(&raw_data);
    }

    stream.closeStream();
    */

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}
