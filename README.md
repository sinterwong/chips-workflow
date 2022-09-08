# Flowengine

That is a temp description.

# Prerequisites
- CMake 3.1+ installed.
- Use a C++17 standard.
- protobuf 3.20+ installed.
- opencv 4.2+ installed.
- openssl installed.

# Build & Install
```bash
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
## Run flow
  ```bash
  cd ${INSTALL_ROOT}/bin
  ./app_flowengine --config_path=your_config.json --num_workers=16
  ```
