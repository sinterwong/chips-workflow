rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/home/wangxt/softwares/flowengine \
      -DCMAKE_BUILD_TYPE=Debug ..

make -j 4

# make install

# ./sample_post_config 9999 "http://114.242.23.39:9400/v1/internal/abutment" /home/wangxt/workspace/projects/flowengine/tests/data/output.json
