FROM nvcr.io/nvidia/tritonserver:24.08-py3 AS base_image
ENV WORKDIR=/workspace/mlops-sandbox
ENV PYTHONPATH /workspace/mlops-sandbox

RUN apt-get update \
    &&  apt-get install -y \
    python-is-python3 \
    pkg-config \
    python3-opencv \
    && apt-get autoremove \
    && apt-get clean

WORKDIR $WORKDIR

# Install OpenCV
# NOTE: OpenCV is installed from pip dees not H.264 support.
RUN apt-get update  && apt install -y \
    build-essential cmake git python3-dev python3-numpy \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev libgtk-3-dev \
    libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev \
    libopencv-dev x264 libx264-dev libssl-dev ffmpeg \
    && apt-get autoremove \
    && apt-get clean \
    && python -m pip install --no-binary opencv-python opencv-python

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
RUN opentelemetry-bootstrap -a install
