# Flow engine

That is a temp description.

# Prerequisites
- CMake 3.1+ installed.
- Use a C++14 compiler(C++11 is optional).
- boost 1.7+ installed.
- protobuf 3.20+ installed.
- opencv 4.2+ installed.
- fmt installed.
- freetype installed.
- openssl installed.

# Build & Install
```bash
$ git clone http://39.101.134.231:8090/core/flowengine.git
$ cd flowengine
$ ln -s ${your_path}/protobuf 3rdparty/target/linux_${your_arch}/protobuf
$ ...
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ make install DESTDIR={your_destination} (optional)
```
# Test
## Startup config server
  ```bash
  $ cd build/${your_arch}/bin # cd build/aarch64/bin
  $ ./sample_post_config ${url} # ./sample_post_config http://localhost:9400/abutment
  ```
## Run flow
  ```bash
  $ ./sample_flow ${id} ${config_path} # ./sample_flow 9999 data/config.json
  ```
  