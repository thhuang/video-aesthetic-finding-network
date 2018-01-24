#!/bin/sh

PRJ_ROOT='/data/thhuang/video-aesthetic-finding-network'
INPUT_PATH='/nas2/thhuang/video-aesthetic-finding-network_input'
OUTPUT_PATH='/nas2/thhuang/video-aesthetic-finding-network_output'
IMAGE_NAME='thhuang/video-aesthetic-finding-network'
PORT='3648'

nvidia-docker run -ti -p ${PORT}:8888 \
                  -v ${PRJ_ROOT}:/app \
                  -v ${INPUT_PATH}:/app/data/input:ro \
                  -v ${OUTPUT_PATH}:/app/data/output \
                  ${IMAGE_NAME} bash

