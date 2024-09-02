mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" ./matmul-allreduce > matmul-results-2-gpus-16-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" ./matmul-allreduce > matmul-results-4-gpus-16-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" ./matmul-allreduce > matmul-results-8-gpus-16-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" ./matmul-allreduce > matmul-results-16-gpus-16-channels.txt
