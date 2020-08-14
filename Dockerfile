FROM nvidia/cuda:11.0-devel

RUN apt-get update -qq

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y build-essential git cmake autoconf libtool pkg-config postgresql-client iproute2

# lib vips
RUN apt-get -y install libvips libvips-dev

#lib curl
RUN apt-get -y install libcurl4-openssl-dev

#lib pqxx
RUN apt-get -y install libpqxx-dev

# lib hiredis and redis-plus-plus
RUN apt-get -y install libhiredis-dev

RUN apt-get -q clean

RUN git clone https://github.com/sewenew/redis-plus-plus.git &&\
  cd redis-plus-plus && mkdir compile &&\
  cd compile &&\
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/lib/redis-plus-plus .. &&\
  make && make install && cd ../.. &&\
  rm -r redis-plus-plus

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# App
WORKDIR /usr/src/app
COPY . . 
#  -Xptxas="-v" 
RUN nvcc data_image_cuda.cu \
  -Iinc -I/usr/lib/redis-plus-plus/include\
  /usr/lib/redis-plus-plus/lib/libredis++.a \
  -o data_image -lm -lstdc++ -lpqxx -lcurl \
  -Xcompiler="$(pkg-config vips --cflags --libs)" -lhiredis \
  -lvips -lz -lgobject-2.0 \
  -lgmodule-2.0 -lgthread-2.0 -lglib-2.0
# RUN nvcc data_image_cuda.cu -o data_image_cuda 
# Start the main process.
CMD bash -c "./data_image"