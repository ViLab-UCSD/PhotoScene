#!/bin/bash

yamlFile="${1:-"configs/total3d/Total_246.yaml"}"

### Initialization and Alignment ###
cmd="python3 photoscene/preprocess.py --config $yamlFile"
echo $cmd
eval $cmd

### Graph Selection ###
cmd="python3 photoscene/selectGraphFromCls.py --config $yamlFile"
echo $cmd
eval $cmd

### First Round Material Optimization ###
cmd="python3 photoscene/optimizeMaterial.py --config $yamlFile --mode first"
echo $cmd
eval $cmd

### Lighting Optimization ### 
cmd="python3 photoscene/optimizeLight.py --config $yamlFile"
echo $cmd
eval $cmd

### Second Round Material Optimization ###
cmd="python3 photoscene/optimizeMaterial.py --config $yamlFile --mode second"
echo $cmd
eval $cmd

# ### Render Final PhotoScene Result ###
cmd="python3 photoscene/renderPhotoScene.py --config $yamlFile"
echo $cmd
eval $cmd
