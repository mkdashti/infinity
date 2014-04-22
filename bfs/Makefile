CUDA_DIR := /usr/local/cuda-6.0
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

CC := $(CUDA_DIR)/bin/nvcc -g -G -arch=sm_30

INCLUDE := $(CUDA_DIR)/include

OBJ = bfs.o infalloc.o

all: bfs

bfs : $(OBJ)
	$(CC) $(OBJ) $(CC_FLAGS) -o $@
bfs.o: bfs.cu kernel.cu kernel2.cu
	$(CC) -c  $< $(CC_FLAGS)
infalloc.o: infalloc.cu infalloc.h
	$(CC) -c  $< $(CC_FLAGS)

clean: 
	rm -f bfs *.o

