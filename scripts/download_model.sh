#!/bin/bash

# Download Total3D model weight
mkdir -p third_party/Total3D/out/pretrained_models
# Download pretrained model pretrained_model.pth from Total3D (288M)
if [ ! -f third_party/Total3D/out/pretrained_models/pretrained_model.pth ]; then
    wget https://tumde-my.sharepoint.com/:u:/g/personal/yinyu_nie_tum_de/EZ0L-aDmX9hMh32-N99zby4Bn4iwI37rsr32VvDt1u1uzg?download=1 \
        -O third_party/Total3D/out/pretrained_models/pretrained_model.pth
fi

# Download pretrained Mesh Generation Net meshnet_model.pth from Total3D (65M)
if [ ! -f third_party/Total3D/out/pretrained_models/meshnet_model.pth ]; then
    wget https://tumde-my.sharepoint.com/:u:/g/personal/yinyu_nie_tum_de/EW7QpZIGUkxGtUdBwkPi9BQBn1oODQKPb08uuFI1LiSoNw?download=1 \
        -O third_party/Total3D/out/pretrained_models/meshnet_model.pth
fi

# Download Maskformer model weight (245M)
if [ ! -f third_party/MaskFormer/demo/model_final_0b3cc8.pkl ]; then
    wget https://dl.fbaipublicfiles.com/maskformer/panoptic-ade20k/maskformer_panoptic_R101_bs16_720k/model_final_0b3cc8.pkl \
        -P third_party/MaskFormer/demo
fi

# Download inverse rendering net model weight (1.2G)
if [ ! -f third_party/InverseRenderingOfIndoorScene/models.zip ]; then
    wget http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip -P third_party/InverseRenderingOfIndoorScene
    unzip third_party/InverseRenderingOfIndoorScene/models.zip -d third_party/InverseRenderingOfIndoorScene
    cd third_party/InverseRenderingOfIndoorScene
    mv models/* .
    rm -r models
    cd ../..
fi
