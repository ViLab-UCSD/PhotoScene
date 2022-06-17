#!/bin/bash

# Download inverse rendering net model weight
if [ ! -f third_party/InverseRenderingOfIndoorScene/models.zip ]; then
    wget http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip -P third_party/InverseRenderingOfIndoorScene
    unzip third_party/InverseRenderingOfIndoorScene/models.zip -d third_party/InverseRenderingOfIndoorScene
fi