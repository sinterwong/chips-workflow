scp -P 22 -r build/aarch64/bin \
               build/aarch64/lib \
               conf \
               tests/data/data \
               tests/data/models \
               root@192.168.0.188:/root/workspace/softwares/flowengine

scp -P 9205 -r build/aarch64/bin \
               build/aarch64/lib/*.so* \
               conf/app \
               tests/data/models \
               root@114.242.23.39:/root/workspace/softwares/flowengine

scp -P 22 -r build/aarch64/bin \
               build/aarch64/lib \
               conf \
               tests/data/models \
               root@192.168.31.216:/root/workspace/softwares/flowengine

scp -P 22 -r  build/aarch64/bin/test_face \
              root@192.168.31.216:/root/workspace/softwares/flowengine