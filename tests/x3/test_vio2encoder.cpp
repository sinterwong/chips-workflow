#include "sp_codec.h"
#include "sp_display.h"
#include "sp_sys.h"
#include "sp_vio.h"

#include <bits/types/FILE.h>
#include <cstring>
#include <iostream>
#include <signal.h>

#include <atomic>
#include <filesystem>
#include <gflags/gflags.h>

DEFINE_int32(output_width, 1920, "Specify the width for video.");
DEFINE_int32(output_height, 1080, "Specify the width for video.");
DEFINE_string(output_path, "out",
              "Specify the path without postfix for output.");
DEFINE_bool(display, false, "Whether to display on the screen.");

using std::printf;

#define STREAM_FRAME_SIZE 2097152
static std::string doc =
    "vio2encode sample -- An example of using the camera to record and encode";

std::atomic_bool is_stop;
void signal_handler_func(int signum) {
  printf("\nrecv:%d,Stoping...\n", signum);
  is_stop.store(true);
}

void close(FILE *stream, char *stream_buffer, void *encoder, void *vio_object) {
  /* file close*/
  fclose(stream);
  /*head memory release*/
  free(stream_buffer);
  /*stop module*/
  sp_stop_encode(encoder);
  sp_vio_close(vio_object);
  /*release object*/
  sp_release_encoder_module(encoder);
  sp_release_vio_module(vio_object);
}

void wrapH2642mp4(std::string const &h264File, std::string const &mp4File) {
  if (!std::filesystem::exists(h264File)) {
    return;
  };
  std::string cmd = "ffmpeg -i " + h264File + " -c:v copy " + mp4File;
  int ret = std::system(cmd.c_str());
  if (ret == -1) {
    perror("system");
    std::exit(EXIT_FAILURE);
  } else {
    if (WIFEXITED(ret)) {
      WEXITSTATUS(ret);
    } else if (WIFSIGNALED(ret)) {
      WTERMSIG(ret);
    }
  }
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // singal handle,stop program while press ctrl + c
  signal(SIGINT, signal_handler_func);
  int ret = 0, i = 0;
  int stream_frame_size = 0;
  int width = FLAGS_output_width, height = FLAGS_output_height;
  // init module
  void *vio_object = sp_init_vio_module();
  void *encoder = sp_init_encoder_module();
  char *stream_buffer =
      reinterpret_cast<char *>(malloc(sizeof(char) * STREAM_FRAME_SIZE));

  FILE *stream = fopen(FLAGS_output_path.c_str(), "wb+");

  // open camera
  ret = sp_open_camera(vio_object, 0, 1, &width, &height);
  if (ret != 0) {
    printf("[Error] sp_open_camera failed!\n");
    close(stream, stream_buffer, encoder, vio_object);
    return -1;
  }
  printf("sp_open_camera success!\n");
  // begin encode
  ret = sp_start_encode(encoder, 0, SP_ENCODER_H264, width, height, 8000);
  if (ret != 0) {
    printf("[Error] sp_start_encode failed!\n");
    close(stream, stream_buffer, encoder, vio_object);
    return -1;
  }
  printf("sp_start_encode success!\n");
  // bind camera(vio) and decoder
  ret = sp_module_bind(vio_object, SP_MTYPE_VIO, encoder, SP_MTYPE_ENCODER);
  if (ret != 0) {
    printf("sp_module_bind(vio -> encoder) failed\n");
    close(stream, stream_buffer, encoder, vio_object);
    return -1;
  }
  printf("sp_module_bind(vio -> encoder) success!\n");

  if (FLAGS_display) {
    // 获取显示器支持的分辨率
    int disp_w = 0, disp_h = 0;
    sp_get_display_resolution(&disp_w, &disp_h);
    printf("disp_w=%d, disp_h=%d\n", disp_w, disp_h);
    // display
    void *display_obj = sp_init_display_module();
    // 使用通道1，这样不会破坏图形化系统，在程序退出后还能恢复桌面
    ret = sp_start_display(display_obj, 1, disp_w, disp_h);
    if (ret != 0) {
      printf("sp_start_display failed\n");
      sp_stop_display(display_obj);
      sp_release_display_module(display_obj);
      close(stream, stream_buffer, encoder, vio_object);
      return -1;
    }
    ret =
        sp_module_bind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
    if (ret != 0) {
      printf("sp_module_bind(vio -> display) failed\n");
      sp_stop_display(display_obj);
      sp_release_display_module(display_obj);
      close(stream, stream_buffer, encoder, vio_object);
      return -1;
    }
    printf("sp_module_bind(vio -> display) success!\n");
  }

  while (!is_stop.load()) {
    memset(stream_buffer, 0, STREAM_FRAME_SIZE);
    // get stream from encoder
    stream_frame_size = sp_encoder_get_stream(encoder, stream_buffer);
    // printf("size:%d\n", stream_frame_size);
    if (stream_frame_size == -1) {
      printf("encoder_get_image error! ret = %d,i = %d\n", ret, i++);
      close(stream, stream_buffer, encoder, vio_object);
      return -1;
    }
    // write stream to file
    fwrite(stream_buffer, sizeof(char), stream_frame_size, stream);
  }

  wrapH2642mp4(FLAGS_output_path, FLAGS_output_path + ".mp4");

  std::error_code ec;
  int retval = std::filesystem::remove(FLAGS_output_path, ec);
  if (!ec) { // Success
    printf("successful: \n");
    if (retval) {
      printf("cache existed and removed\n");
    } else {
      printf("cache didn't exist\n");
    }
  } else { // Error
    printf("unsuccessful: %d %s \n", ec.value(), ec.message().c_str());
  }
  gflags::ShutDownCommandLineFlags();

  return 0;
}
