# Data Image Cuda

## Description
This Program moves the heavy computation of building the data images into CUDA

## Benchmark
On a dataset of 4371095 vectors, and an image of 256x256, this program takes 2106.425 msec, on the lowest tier Google Cloud GPU (Tesla T4) with a low tier cpu.
On heroku in console using the old method (still running c code through a rubby extension), running a hobby cpu, this same computation took 1079.774328292 seconds.