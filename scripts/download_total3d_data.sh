#!/bin/bash

# Download preprocessed SUN RGBD data from Total3D (12G)
wget https://tumde-my.sharepoint.com/:u:/g/personal/yinyu_nie_tum_de/Ef_cwoqSdA1DlpjcEk8PAfIBiutQTW8w6yRTlBJoGRxy5w?download=1 \
    -O third_party/Total3D/data/sunrgbd/sunrgbd_train_test_data.tar.gz
tar -xf third_party/Total3D/data/sunrgbd/sunrgbd_train_test_data.tar.gz -C third_party/Total3D/data/sunrgbd
