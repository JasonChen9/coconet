include ../common.mk

all: adam adam-ar-c adam-rs-c-ag adam-fuse-rs-c-ag

adam: adam.cpp
	g++ $(COCONET_FLAGS) $< -g -O0 -o $@ -std=c++14

adam-ar-c.cu: adam
	./adam AR_C

adam-rs-c-ag.cu: adam
	./adam RS_C_AG

adam-fuse-rs-c-ag.cu: adam
	./adam fuse_RS_C_AG

adam-ar-c: adam-ar-c.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I$(CUDA_PATH)/include/ -I../../ -I$(NCCL_OVERLAP_BUILD_PATH)/include -L$(NCCL_OVERLAP_BUILD_PATH)/lib/ -L$(CUDA_PATH)/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_OVERLAP_BUILD_PATH)/lib -L$(CUDA_PATH)/lib64/ -L/mnt/sdb/xiangguangyu/code/coconet/nccl/lib -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

adam-rs-c-ag: adam-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I$(CUDA_PATH)/include/ -I../../ -I$(NCCL_OVERLAP_BUILD_PATH)/include -L$(NCCL_OVERLAP_BUILD_PATH)/lib/ -L$(CUDA_PATH)/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_OVERLAP_BUILD_PATH)/lib -L$(CUDA_PATH)/lib64/ -L/mnt/sdb/xiangguangyu/code/coconet/nccl/lib -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

adam-fuse-rs-c-ag: adam-fuse-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I$(CUDA_PATH)/include/ -I../../ -I$(NCCL_OVERLAP_BUILD_PATH)/include -L$(NCCL_OVERLAP_BUILD_PATH)/lib/ -L$(CUDA_PATH)/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_OVERLAP_BUILD_PATH)/lib -L$(CUDA_PATH)/lib64/ -L/mnt/sdb/xiangguangyu/code/coconet/nccl/lib -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

clean:
	rm -rf adam adam-ar-c.cu adam-ar-c adam-rs-c-ag.cu adam-rs-c-ag adam-fuse-rs-c-ag.cu adam-fuse-rs-c-ag
