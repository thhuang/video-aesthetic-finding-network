#!/bin/sh

PRJ_ROOT='/storage/data/thhuang/video-aesthetic-finding-network'
INPUT_PATH='/storage/data/thhuang/video-aesthetic-finding-network_input'
OUTPUT_PATH='/storage/data/thhuang/video-aesthetic-finding-network_output'
DOWNLOADS_PATH='/storage/data/thhuang/video-aesthetic-finding-network_downloads'
IMAGE_NAME='hc1.corp.ailabs.tw:6000/video-aesthetic-finding-network'
PORT='6567'

nvidia-docker run -ti -p ${PORT}:8888 \
                  -v ${PRJ_ROOT}:/app \
                  -v ${INPUT_PATH}:/app/data/input:ro \
                  -v ${OUTPUT_PATH}:/app/data/output \
                  -v ${DOWNLOADS_PATH}:/app/data/downloads \
                  ${IMAGE_NAME} bash

