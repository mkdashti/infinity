#!/bin/bash
nvcc -arch=sm_30 -O3 -I ~/usr/include -dc bpt.cu -o bpt.o
nvcc -arch=sm_30 -O3 -L ~/usr/lib -lhalloc -o bpt bpt.o
