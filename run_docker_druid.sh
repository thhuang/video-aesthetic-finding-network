#!/bin/sh

PRJ_ROOT='/data/thhuang/video-aesthetic-finding-network'
INPUT_PATH='/data/thhuang/video-aesthetic-finding-network_input'
OUTPUT_PATH='/data/thhuang/video-aesthetic-finding-network_output'
DOWNLOADS_PATH='/data/thhuang/video-aesthetic-finding-network_downloads'
IMAGE_NAME='thhuang/video-aesthetic-finding-network'
PORT='3684'

nvidia-docker run -ti -p ${PORT}:8888 \
                  -v ${PRJ_ROOT}:/app \
                  -v ${INPUT_PATH}:/app/data/input:ro \
                  -v ${OUTPUT_PATH}:/app/data/output \
                  -v ${DOWNLOADS_PATH}:/app/data/downloads \
                  ${IMAGE_NAME} bash

