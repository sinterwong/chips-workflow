# chips-workflow

## Introduction
chips-workflow is a framework for video processing. It is designed for edge computing and supports multiple platforms. It is based on the pipeline model and supports the following functions:
- video decoding
- video encoding
- image inference
- video pipeline inference 

## Denpendencies
- cmake >= 3.10
- g++ >= 7.5
- spdlog
- gflags
- opencv >= 4.2
- onnxruntime >= 1.16.3
- ffmepeg >= 6.1(when using video)
- curl(when using module or server)
- openssl(when using license)
- oatpp >= 1.3.0(when using server)
- faiss(when using servre/face)
- jetson-utils(when using video in jetson platform)

## Project Structure
```bash
├── 3rdparty
│   ├── ...
│   └── target
│       ├── linux_aarch64
│       │   ├── dependency1
│       │   └── dependency2
│       └── linux_x86_64
├── build(build files)
├── app(applications)
├── cmake(cmake files)
├── server
├── src
│   ├── common
│   ├── infer
│   ├── license
│   ├── module
│   ├── logger
│   ├── utils
│   └── video
└── tests(test cases)
```

## Build
### dependencies setting
Please see the cmake/load_3rdparty.cmake file for needed dependencies in different platforms.

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
    "-DAPPSDK_PATH=/path/root"  // for some platforms which need sdk like horizon-x3pi
  ],
  "cmake.cacheInit": null,
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "git.terminalAuthentication": true
}

```

## Run
```bash
cd build/your_platform/bin
./your_app --args
```

## TODO
- [x] support generic platform with onnxruntime(x86_64, aarch64)
- [x] support soft-decoding with ffmpeg
- [x] support Jetson
- [x] support Horizon X3pi
- [x] pipeline inference demo(app/app_flowengine.cpp)
- [x] face recognition server
- [ ] support Rockchip RK3588
- [ ] pipeline inference server
