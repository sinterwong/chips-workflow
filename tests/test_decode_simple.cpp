//
// Created by xl on 2022/1/7.
//

#include <iostream>
#include <string>

// #include "rk_mpi.h"

// #include "mpp_log.h"
// #include "mpp_mem.h"
// #include "mpp_env.h"
// #include "mpp_time.h"
// #include "mpp_common.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}
using namespace std;

int OpenRtspStream(const char *url, AVFormatContext **ic) {
  AVFormatContext *pFormatCtx = *ic;
  AVDictionary *options = NULL;
  int ret = -1;
  // avformat_close_input 关闭
  if (avformat_open_input(ic, url, NULL, &options) != 0) {
    if (!(*ic)) {
      std::cout << "Video closed" << std::endl;
      avformat_free_context(*ic);
    }
    return -1;
  }
  if (avformat_find_stream_info(*ic, NULL) < 0) {
    if (!(*ic)) {
      avformat_close_input(ic);
      avformat_free_context(*ic);
    }
    return -1;
  }

  printf("-----------rtsp流输入信息--------------\n");
  av_dump_format(*ic, 0, url, 0);
  printf("---------------------------------------\n");
  printf("\n");

  // find video stream
  int videoindex = -1;
  for (int i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoindex = i;
      break;
    }
  }
  if (videoindex == -1) {
    printf("Didn't find a video stream.\n");
    return -1;
  }
  auto pCodecCtx = pFormatCtx->streams[videoindex]->codec;
  //    auto i_video_stream = pFormatCtx->streams[videoindex];

  // decoder
  auto pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
  if (pCodec == NULL) {
    printf("Codec not found.\n");
    return -1;
  }

  // Step 6: open the decoder
  // Set cache size 1024000byte s
  av_dict_set(&options, "buffer_size", "1024000", 0);
  // Set the timeout for 20s
  av_dict_set(&options, "stimeout", "20000", 0);
  // Set the maximum delay of 3s
  av_dict_set(&options, "max_delay", "30000", 0);
  // Setting the open mode tcp/udp
  av_dict_set(&options, "rtsp_transport", "tcp", 0);

  if (avcodec_open2(pCodecCtx, pCodec, &options) < 0) {
    printf("Could not open codec.\n");
    return -1;
  }

  std::cout << "Bit rate:" << pCodecCtx->bit_rate << std::endl;
  std::cout << "Width and height:" << pCodecCtx->width << "x"
            << pCodecCtx->height << std::endl;
  // AV_PIX_FMT_YUV420P 0
  std::cout << "format:" << pCodecCtx->pix_fmt << std::endl;
  std::cout << "Frame rate denominator:" << pCodecCtx->time_base.den
            << std::endl;

  auto packet = (AVPacket *)av_malloc(sizeof(AVPacket));

  //    unsigned char _buf[2000000];
  //   auto _buf = mpp_malloc(char, 2000000);
  //   if (NULL == _buf) {
  //     std::cout << "[ERROR] mpi_dec_test malloc input stream buffer failed"
  //               << std::endl;
  //     return -1;
  //   }

  int id = 0;
  int offset = 0;
  bool start_flag = false;
  int num_key_frame = 0;
  while (true) {
    av_init_packet(packet);
    if (av_read_frame(pFormatCtx, packet) >= 0) {
			std::cout << "av_read_frame successful!" << std::endl;
      if (packet->stream_index == videoindex) {
        id++;
        std::cout << "current ID:" << id << ", packet flags:" << packet->flags
                  << ", packet size: " << packet->size << std::endl;

        // START FROM THE SECOND KEY FRAME !
        if (packet->flags == 1 && !start_flag) {
          num_key_frame++;
          if (num_key_frame == 2) {
            start_flag = true;
            //                        memcpy(DecCfg.buf,
            //                        AVcfg.pCodecCtx->extradata,
            //                        AVcfg.pCodecCtx->extradata_size);
            offset += pCodecCtx->extradata_size;
            //                        memcpy(DecCfg.buf + offset,
            //                        AVcfg.packet->data, AVcfg.packet->size);
            offset += packet->size;
          }
        } else if (start_flag) {
          //   memcpy(_buf, packet->data, packet->size);
          std::cout << "Copy video data here!" << std::endl;
        } // else continue
      }
      av_packet_unref(packet);
    } else {
        std::cout << "frame read failed!" << std::endl;
        continue;
    }
  }

  avcodec_close(pCodecCtx);
  avformat_close_input(&pFormatCtx);

  //   mpp_free(_buf);

  return 0;
}

int main(int argc, char **argv) {
  //    std::string stream="rtsp://192.168.101.112:8554/ch01.264?dev=1";
//   std::string stream = "rtsp://admin:zkfd123.com@114.242.23.39:9302/cam/"
//                        "realmonitor?channel=1&subtype=0";
	std::string stream = "rtsp://admin:zkfd123.com@114.242.23.39:9304/Streaming/Channels/101";

  AVFormatContext *ifmt = NULL;
  avcodec_register_all();
  av_register_all();
  avformat_network_init();
  ifmt = avformat_alloc_context();
  if (!ifmt) {
    cout << "avformatcontext alloc error" << endl;
    return -1;
  }
  auto ret = OpenRtspStream(stream.c_str(), &ifmt);
  if (ret < 0) {
    cout << "摄像机网络不通" << endl;
    return -1;
  }
  return 0;
}

// int main(int argc, char **argv) {
//     return 0;
// }