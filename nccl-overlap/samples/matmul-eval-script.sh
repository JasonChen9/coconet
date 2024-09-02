mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2  -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-2-gpus-2-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2  -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-4-gpus-2-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2  -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-8-gpus-2-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2  -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-16-gpus-2-channels.txt

mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-2-gpus-4-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-4-gpus-4-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-8-gpus-4-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-16-gpus-4-channels.txt

mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-2-gpus-8-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-4-gpus-8-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-8-gpus-8-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-16-gpus-8-channels.txt

mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-2-gpus-12-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-4-gpus-12-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-8-gpus-12-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-16-gpus-12-channels.txt

mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-2-gpus-16-channels.txt
mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-4-gpus-16-channels.txt
mpirun -np 8 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-8-gpus-16-channels.txt
mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib:/mnt/sdb/xiangguangyu/opt/CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH" -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 -x NCCL_NTHREADS=512 -x NCCL_BUFFSIZE=4194304 ./matmul-allreduce > matmul-results-16-gpus-16-channels.txt
