scp -P 9202 -r build/aarch64/bin \
               build/aarch64/include \
               build/aarch64/lib \
               conf \
               tests/data/data \
               tests/data/models \
               root@114.242.23.39:/root/flowengine
# scp -P 6322 -r build/aarch64/bin tests/models tests/datas sunrise@39.107.43.174:/home/sunrise/basic-x3
