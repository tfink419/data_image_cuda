FROM nvidia/cuda:11.0-devel

RUN apt-get update -qq

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y build-essential git cmake autoconf libtool pkg-config postgresql-client

# lib vips
RUN apt-get -y install libvips libvips-dev

#lib curl
RUN apt-get -y install libcurl4-openssl-dev

#lib pqxx
RUN apt-get -y install libpqxx-dev

RUN apt-get -q clean

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ARG PORT=8080
ENV PORT="${PORT}"
ARG DATA_IMAGE_CUDA_SECRET_KEY=hellothere
ENV DATA_IMAGE_CUDA_SECRET_KEY="${DATA_IMAGE_CUDA_SECRET_KEY}"

EXPOSE ${PORT}

# App
WORKDIR /usr/src/app
COPY . .
RUN gcc data_image.cpp -o data_image -lm -lstdc++ -lpqxx -lcurl `pkg-config vips --cflags --libs`
# RUN nvcc data_image_cuda.cu -o data_image_cuda -I inc
# Start the main process.
CMD bash -c "./data_image $PORT $DATA_IMAGE_CUDA_SECRET_KEY"
# CMD "./data_image_cuda"
# CMD "./data_image_cuda -p $PORT"