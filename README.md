# chips-workflow

## Description
chips-workflow is a dnn inference workflow engine that allows you to create and run pipeline for video and picture scenes.

## Denpendencies
- C++17
- CMake 3.10 or higher
- spdlog
- gflags
- ffmpeg
- opencv 4.2 or higher
- Eigen 3+
- curl
- openssl

## Sturcture
- src
  - common: common code
  - core: core backend code
  - infer: algorithms and dnnEngine wrapper for inference
  - license: license code
  - logger: logger
  - modules: function modules
  - utils: utilities
  - video: video processing and platforms enc and dec wrapper
- tests: test code
- app: application code
- 3rdparty: third party libraries
- cmake: environment construction related files.
- scripts: scripts for building and deploying etc.
- platform: cross-compile related files

## Build
### dependencies setting
```bash
cd 3rdparty/target/linux_${platform}/
ln -s ${your_libname_path} libname
```

### command line
```bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../platform/toolchain/aarch64.platform.cmake -DTOOLCHAIN_ROOTDIR=/path/root ..
```
### vscode cmake tools(example)
```json
{
  "cmake.installPrefix": "/path/softwares/flowengine",
  "cmake.configureArgs": [
    "-DCMAKE_TOOLCHAIN_FILE=${workspaceFolder}/platforms/linux/aarch64.platform.cmake",
    "-DTOOLCHAIN_ROOTDIR=/opt/gcc-aarch64-linux-gnu/bin",
    "-DAPPSDK_PATH=/path/root"
  ],
  "cmake.cacheInit": null,
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "git.terminalAuthentication": true
}

```

## Run
```bash
cd build/aarch64/bin
./your_app --args
```

## TODO
- [x] support Jetson NX
- [x] support Horizon X3pi
- [ ] support Rockchip RK3588
- [ ] gtest unit test
- [ ] optimize concurrency for infer
- [ ] http server for pipeline
