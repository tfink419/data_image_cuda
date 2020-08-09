FROM nvidia/cuda:11.0-devel

RUN apt-get update -qq

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y build-essential git cmake autoconf libtool pkg-config postgresql-client
# lib vips
RUN apt-get -y install libvips libvips-dev
RUN apt-get -q clean

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# App
WORKDIR /usr/src/app
COPY . .
#RUN gcc data_image.cpp -o data_image -lm -lstdc++
RUN nvcc data_image_cuda.cu -o data_image_cuda -I inc
# Start the main process.
CMD "./data_image_cuda -p $PORT"