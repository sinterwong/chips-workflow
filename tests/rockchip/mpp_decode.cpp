#include "ffstream.hpp"
#include "gflags/gflags.h"
#include "logger/logger.hpp"
#include "rk_mpi.h"
#include <string.h>

#include "osal/mpp_common.h"
#include "osal/mpp_env.h"
#include "osal/mpp_mem.h"
#include "osal/mpp_time.h"

#include <opencv2/opencv.hpp>
#include <thread>

#define MPI_DEC_STREAM_SIZE 1920 * 1080 * 4

DEFINE_string(uri, "", "Specify the uri of video.");

typedef struct {
  MppCtx ctx;
  MppApi *mpi;

  /* end of stream flag when set quit the loop */
  RK_U32 eos;

  /* buffer for stream data reading */
  char *buf;

  /* input and output */
  MppBufferGroup frm_grp;
  MppBufferGroup pkt_grp;
  MppPacket packet;
  size_t packet_size;
  MppFrame frame;
  RK_S32 frame_count;
  RK_S32 frame_num;
  size_t max_usage;
} MpiDecLoopData;

static int decode_simple(MpiDecLoopData *data) {
  RK_U32 pkt_eos = 0;
  MPP_RET ret = MPP_OK;
  MppCtx ctx = data->ctx;
  MppApi *mpi = data->mpi;
  void *buf = reinterpret_cast<void *>(data->buf);
  MppPacket packet = data->packet;
  MppFrame frame = nullptr;

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

  while (stream.isRunning() && !pkt_eos) {
    int bufSize = stream.getRawFrame(&buf, false);
    // setup eos flag
    if (bufSize < 0) {
      mpp_packet_set_eos(packet);
      pkt_eos = 1;
    } else {
      mpp_packet_set_data(packet, buf);
      mpp_packet_set_size(packet, bufSize);
      mpp_packet_set_pos(packet, buf);
      mpp_packet_set_length(packet, bufSize);
    }

    // send the packet first if packet is not done
    ret = mpi->decode_put_packet(ctx, packet);
    if (MPP_OK != ret) {
      FLOWENGINE_LOGGER_ERROR("{} decode_put_packet failed ret {}", ctx, ret);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));

    RK_S32 times = 5; // 超时尝试次数
    // then get all available frame and release
    while (true) {
      RK_U32 frm_eos = 0;

      ret = mpi->decode_get_frame(ctx, &frame);
      if (MPP_ERR_TIMEOUT == ret) {
        if (times > 0) {
          times--;
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        FLOWENGINE_LOGGER_ERROR("{} decode_get_frame failed too much time",
                                ctx);
        continue;
      }
      if (MPP_OK != ret) {
        FLOWENGINE_LOGGER_ERROR("{} decode_get_frame failed ret {}", ctx, ret);
        break;
      }

      if (frame) {
        if (mpp_frame_get_info_change(frame)) {
          RK_U32 width = mpp_frame_get_width(frame);
          RK_U32 height = mpp_frame_get_height(frame);
          RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
          RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
          RK_U32 buf_size = mpp_frame_get_buf_size(frame);
          MppFrameFormat fmt = mpp_frame_get_fmt(frame);
          FLOWENGINE_LOGGER_INFO("{} decode_get_frame get info changed found",
                                 ctx);
          FLOWENGINE_LOGGER_INFO("{} decoder require buffer w:h [{}:{}] stride "
                                 "[{}:{}] buf_size {}, fmt {}",
                                 ctx, width, height, hor_stride, ver_stride,
                                 buf_size, fmt);
          /*
           * NOTE: We can choose decoder's buffer mode here.
           * There are three mode that decoder can support:
           *
           * Mode 1: Pure internal mode
           * In the mode user will NOT call MPP_DEC_SET_EXT_BUF_GROUP
           * control to decoder. Only call MPP_DEC_SET_INFO_CHANGE_READY
           * to let decoder go on. Then decoder will use create buffer
           * internally and user need to release each frame they get.
           *
           * Advantage:
           * Easy to use and get a demo quickly
           * Disadvantage:
           * 1. The buffer from decoder may not be return before
           * decoder is close. So memroy leak or crash may happen.
           * 2. The decoder memory usage can not be control. Decoder
           * is on a free-to-run status and consume all memory it can
           * get.
           * 3. Difficult to implement zero-copy display path.
           *
           * Mode 2: Half internal mode
           * This is the mode current test code using. User need to
           * create MppBufferGroup according to the returned info
           * change MppFrame. User can use mpp_buffer_group_limit_config
           * function to limit decoder memory usage.
           *
           * Advantage:
           * 1. Easy to use
           * 2. User can release MppBufferGroup after decoder is closed.
           *    So memory can stay longer safely.
           * 3. Can limit the memory usage by mpp_buffer_group_limit_config
           * Disadvantage:
           * 1. The buffer limitation is still not accurate. Memory usage
           * is 100% fixed.
           * 2. Also difficult to implement zero-copy display path.
           *
           * Mode 3: Pure external mode
           * In this mode use need to create empty MppBufferGroup and
           * import memory from external allocator by file handle.
           * On Android surfaceflinger will create buffer. Then
           * mediaserver get the file handle from surfaceflinger and
           * commit to decoder's MppBufferGroup.
           *
           * Advantage:
           * 1. Most efficient way for zero-copy display
           * Disadvantage:
           * 1. Difficult to learn and use.
           * 2. Player work flow may limit this usage.
           * 3. May need a external parser to get the correct buffer
           * size for the external allocator.
           *
           * The required buffer size caculation:
           * hor_stride * ver_stride * 3 / 2 for pixel data
           * hor_stride * ver_stride / 2 for extra info
           * Total hor_stride * ver_stride * 2 will be enough.
           *
           * For H.264/H.265 20+ buffers will be enough.
           * For other codec 10 buffers will be enough.
           */

          if (nullptr == data->frm_grp) {
            /* If buffer group is not set create one and limit it */
            ret = mpp_buffer_group_get_internal(&data->frm_grp,
                                                MPP_BUFFER_TYPE_ION);
            if (ret) {
              FLOWENGINE_LOGGER_ERROR("{} get mpp buffer group failed ret {}",
                                      ctx, ret);
              break;
            }

            /* Set buffer to mpp decoder */
            ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, data->frm_grp);
            if (ret) {
              FLOWENGINE_LOGGER_ERROR("{} set buffer group failed ret {}", ctx,
                                      ret);
              break;
            }
          } else {
            /* If old buffer group exist clear it */
            ret = mpp_buffer_group_clear(data->frm_grp);
            if (ret) {
              FLOWENGINE_LOGGER_ERROR("{} clear buffer group failed ret {}",
                                      ctx, ret);
              break;
            }
          }

          /* Use limit config to limit buffer count to 24 with buf_size */
          ret = mpp_buffer_group_limit_config(data->frm_grp, buf_size, 24);
          if (ret) {
            FLOWENGINE_LOGGER_ERROR("{} limit buffer group failed ret {}", ctx,
                                    ret);
            break;
          }

          /*
           * All buffer group config done. Set info change ready to let
           * decoder continue decoding
           */
          ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
          if (ret) {
            FLOWENGINE_LOGGER_ERROR("{} info change ready failed ret {}", ctx,
                                    ret);
            break;
          }
        } else {
          char log_buf[256];
          RK_S32 log_size = sizeof(log_buf) - 1;
          RK_S32 log_len = 0;
          RK_U32 err_info = mpp_frame_get_errinfo(frame);
          RK_U32 discard = mpp_frame_get_discard(frame);

          log_len += snprintf(log_buf + log_len, log_size - log_len,
                              "decode get frame %d", data->frame_count);

          if (mpp_frame_has_meta(frame)) {
            MppMeta meta = mpp_frame_get_meta(frame);
            RK_S32 temporal_id = 0;

            mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id);

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                                " tid %d", temporal_id);
          }

          if (err_info || discard) {
            log_len += snprintf(log_buf + log_len, log_size - log_len,
                                " err %x discard %x", err_info, discard);
          }
          FLOWENGINE_LOGGER_INFO("{} {}", ctx, log_buf);
          data->frame_count++;
        }
        { // save image
          RK_U32 width = mpp_frame_get_width(frame);
          RK_U32 height = mpp_frame_get_height(frame);
          RK_U32 h_stride = mpp_frame_get_hor_stride(frame);
          RK_U32 v_stride = mpp_frame_get_ver_stride(frame);
          MppFrameFormat fmt = mpp_frame_get_fmt(frame);
          FLOWENGINE_LOGGER_INFO("{} width {} height {} h_stride {} v_stride "
                                 "{} fmt {}",
                                 ctx, width, height, h_stride, v_stride, fmt);

          // get data buffer
          MppBuffer data_buffer = mpp_frame_get_buffer(frame);
          if (data_buffer != nullptr) {
            RK_U8 *base = (RK_U8 *)mpp_buffer_get_ptr(data_buffer);
            RK_U8 *base_y = base;
            RK_U8 *base_c = base + h_stride * v_stride;

            // 分配图像存储空间（nv12）
            uchar *image_data =
                reinterpret_cast<uchar *>(malloc(width * height * 3 / 2));

            // 拷贝y分量数据到image_data
            for (unsigned int i = 0; i < height; i++, base_y += h_stride) {
              // fwrite(base_y, 1, width, fp);
              memcpy(image_data + i * width, base_y, width);
            }
            // 拷贝uv分量数据到image_data
            for (unsigned int i = 0; i < height / 2; i++, base_c += h_stride) {
              memcpy(image_data + width * height + i * width, base_c, width);
            }

            cv::Mat image(1080 * 3 / 2, 1920, CV_8UC1, image_data);
            cv::imwrite("out_nv12.jpg", image);
            free(image_data);
            image_data = nullptr;
          }

          frm_eos = mpp_frame_get_eos(frame);
          mpp_frame_deinit(&frame);
          frame = nullptr;
        }
      }

      // try get runtime frame memory usage
      if (data->frm_grp) {
        size_t usage = mpp_buffer_group_usage(data->frm_grp);
        if (usage > data->max_usage) {
          data->max_usage = usage;
        }
        // FLOWENGINE_LOGGER_INFO("{} usage {} max {}", ctx, usage,
        //                        data->max_usage);
      }

      // if last packet is send but last frame is not found continue
      if (pkt_eos && !frm_eos) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      if (frm_eos) {
        FLOWENGINE_LOGGER_INFO("{} found last frame", ctx);
        break;
      }

      if (data->frame_num > 0 && data->frame_count >= data->frame_num) {
        data->eos = 1;
        break;
      }
      break;
    }

    if (data->frame_num > 0 && data->frame_count >= data->frame_num) {
      data->eos = 1;
      FLOWENGINE_LOGGER_INFO("reach max frame number", data->frame_count);
      break;
    }
    /*
     * why sleep here:
     * mpi->decode_put_packet will failed when packet in internal queue is
     * full,waiting the package is consumed .Usually hardware decode one
     * frame which resolution is 1080p needs 2 ms,so here we sleep 3ms
     * * is enough.
     */
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  return ret;
}

int main(int argc, char **argv) {
  /*
  ffmpeg -y -c:v h264_rkmpp \
         -rtsp_transport tcp \
         -i rtsp://admin:zkfd123.com@localhost:9303/Streaming/Channels/101 \
         -c copy output.mp4
  */
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  RK_S32 ret = 0;
  RK_U32 width = 0;
  RK_U32 height = 0;
  MppCodingType coding = MPP_VIDEO_CodingAVC; // h264
  RK_S32 frame_num = 0;
  size_t pkt_size = MPI_DEC_STREAM_SIZE;

  // base flow context
  MppCtx ctx = nullptr;
  MppApi *mpi = nullptr;

  // input / output
  MppPacket packet = nullptr;
  MppFrame frame = nullptr;

  MpiCmd mpi_cmd = MPP_CMD_BASE;
  MppParam param = nullptr;
  RK_U32 need_split = 1;

  // resources
  char *buf = nullptr;
  MpiDecLoopData data;

  FLOWENGINE_LOGGER_INFO("mpi_dec_test start");
  mpp_env_set_u32("mpi_debug", 1);

  memset(&data, 0, sizeof(data));

  // buf = mpp_malloc(char, pkt_size);
  // if (nullptr == buf) {
  //   FLOWENGINE_LOGGER_ERROR("mpi_dec_test malloc input stream buffer failed");
  //   goto MPP_TEST_OUT;
  // }

  ret = mpp_packet_init(&packet, buf, pkt_size);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("mpp_packet_init failed");
    goto MPP_TEST_OUT;
  }

  // decoder demo
  ret = mpp_create(&ctx, &mpi);

  if (MPP_OK != ret) {
    FLOWENGINE_LOGGER_ERROR("mpp_create failed");
    goto MPP_TEST_OUT;
  }
  FLOWENGINE_LOGGER_INFO("{} decoder test start w {} h {} coding {}", ctx,
                         width, height, coding);

  // NOTE: decoder split mode need to be set before init
  mpi_cmd = MPP_DEC_SET_PARSER_SPLIT_MODE;
  param = &need_split;
  ret = mpi->control(ctx, mpi_cmd, param);
  if (MPP_OK != ret) {
    FLOWENGINE_LOGGER_ERROR("{} mpi->control failed", ctx);
    goto MPP_TEST_OUT;
  }

  ret = mpp_init(ctx, MPP_CTX_DEC, coding);
  if (MPP_OK != ret) {
    FLOWENGINE_LOGGER_ERROR("{} mpp_init failed", ctx);
    goto MPP_TEST_OUT;
  }

  data.ctx = ctx;
  data.mpi = mpi;
  data.eos = 0;
  data.buf = buf;
  data.packet = packet;
  data.packet_size = pkt_size;
  data.frame = frame;
  data.frame_count = 0;
  data.frame_num = frame_num;
  decode_simple(&data);

  {
    MppDecQueryCfg query;

    memset(&query, 0, sizeof(query));
    query.query_flag = MPP_DEC_QUERY_ALL;
    ret = mpi->control(ctx, MPP_DEC_QUERY, &query);
    if (ret) {
      FLOWENGINE_LOGGER_ERROR("{} mpi->control query failed", ctx);
      goto MPP_TEST_OUT;
    }

    /*
     * NOTE:
     * 1. Output frame count included info change frame and empty eos frame.
     * 2. Hardware run count is real decoded frame count.
     */
    FLOWENGINE_LOGGER_INFO("{} input {} pkt output {} frm decode {} frames",
                           ctx, query.dec_in_pkt_cnt, query.dec_out_frm_cnt,
                           query.dec_hw_run_cnt);
  }

  ret = mpi->reset(ctx);
  if (MPP_OK != ret) {
    FLOWENGINE_LOGGER_ERROR("{} mpi->reset failed", ctx);
    goto MPP_TEST_OUT;
  }

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();

MPP_TEST_OUT:
  if (packet) {
    mpp_packet_deinit(&packet);
    packet = nullptr;
  }

  if (frame) {
    mpp_frame_deinit(&frame);
    frame = nullptr;
  }

  if (ctx) {
    mpp_destroy(ctx);
    ctx = nullptr;
  }

  if (buf) {
    mpp_free(buf);
    buf = nullptr;
  }

  if (data.pkt_grp) {
    mpp_buffer_group_put(data.pkt_grp);
    data.pkt_grp = nullptr;
  }

  if (data.frm_grp) {
    mpp_buffer_group_put(data.frm_grp);
    data.frm_grp = nullptr;
  }
  mpp_env_set_u32("mpi_debug", 0x0);
  return ret;
}
