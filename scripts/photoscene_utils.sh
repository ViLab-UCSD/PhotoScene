#!/bin/bash

load_base_paths () {
    if [ "$REPODIR" = "" ]; then repoDir=$PWD; else repoDir=$REPODIR; fi
    rootConfig=$1
    OptixRoot=$(yq e '.OptixRoot' $rootConfig)
    if [[ $OptixRoot != /* ]]; then OptixRoot=$repoDir/$OptixRoot; fi
    rendererDir=$(yq e '.rendererRoot' $rootConfig)
    if [[ $rendererDir != /* ]]; then rendererDir=$repoDir/$rendererDir; fi
    toolkitRoot=$(yq e '.toolkitRoot' $rootConfig)
    if [[ $toolkitRoot != /* ]]; then toolkitRoot=$repoDir/$toolkitRoot; fi
}

