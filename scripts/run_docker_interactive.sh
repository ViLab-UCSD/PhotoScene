#!/bin/bash

. scripts/photoscene_utils.sh
yamlFile="${1:-"configs/roots.yaml"}"
load_base_paths $yamlFile
gpuId="${2:-0}"

docker run --gpus "device=$gpuId" \
    --rm \
    -it \
    --volume $PWD:$PWD \
    --volume $toolkitRoot:$toolkitRoot \
    --workdir $PWD \
    --env REPO_DIR=$PWD \
    yyeh/photoscene:v1 \
    /bin/bash
