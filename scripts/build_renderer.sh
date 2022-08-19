#!/bin/bash

. scripts/photoscene_utils.sh
yamlFile="${1:-"configs/roots.yaml"}"
load_base_paths $yamlFile

optix_version="5.1.1"

cd $rendererDir/OptixRenderer_MatPart/src/
rm CMakeCache.txt
cmake -D OptiX_INSTALL_DIR=$OptixRoot/NVIDIA-OptiX-SDK-$optix_version-linux64 ../src/
make

cd $rendererDir/OptixRenderer/src/
rm CMakeCache.txt
cmake -D OptiX_INSTALL_DIR=$OptixRoot/NVIDIA-OptiX-SDK-$optix_version-linux64 ../src/
make

cd $rendererDir/OptixRenderer_UVcoord/src/
rm CMakeCache.txt
cmake -D OptiX_INSTALL_DIR=$OptixRoot/NVIDIA-OptiX-SDK-$optix_version-linux64 ../src/
make

cd $rendererDir/OptixRenderer_Light/src/
rm CMakeCache.txt
cmake -D OptiX_INSTALL_DIR=$OptixRoot/NVIDIA-OptiX-SDK-$optix_version-linux64 ../src/
make
