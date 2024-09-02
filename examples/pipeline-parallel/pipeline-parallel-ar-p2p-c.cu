#include "header.h"
void matMul0(int B, int S, int H, half * w, half * in, half * S0, cublasHandle_t cublasHandle, int comm_size, int rank) {
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  CUBLASCHECK(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
    (H), (B*S), DIVUP((H), comm_size), 
    &alpha,     w, CUDA_R_16F, (H), 
    in, CUDA_R_16F, DIVUP((H), comm_size), 
    &beta, S0, CUDA_R_16F, (H), 
    CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
}

__global__ void binOpFunc0(int B, int S, int H, half * r, half * S0, half * S2, int comm_size, int rank) {
  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  S2[i0] = (S0[i0] + r[i0]);
}

void pipeline_parallel(int B, int S, int H, half* w, half* b, half* in, half* r, half* S2, half* S0, float& elapsedTimebinOpFunc0, float& elapsedTimeAllReduce, float& elapsedTimematMul0, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank, cublasHandle_t cublasHandle){
  cudaEvent_t startpipeline_parallel, stoppipeline_parallel;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startpipeline_parallel));
  CUDACHECK(cudaEventCreate(&stoppipeline_parallel));

  CUDACHECK(cudaEventRecord(startpipeline_parallel, stream));
  matMul0(B, S, H, w, in, S0, cublasHandle, comm_size, rank);
  CUDACHECK(cudaEventRecord(stoppipeline_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stoppipeline_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipeline_parallel,stoppipeline_parallel));
  elapsedTimematMul0 += elapsedTime;

  CUDACHECK(cudaEventRecord(startpipeline_parallel, stream));
  ncclAllReduce(S0, S0, (B*S*H), ncclHalf,ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stoppipeline_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stoppipeline_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipeline_parallel,stoppipeline_parallel));
  elapsedTimeAllReduce += elapsedTime;

  CUDACHECK(cudaEventRecord(startpipeline_parallel, stream));
  size_t totalThreads_1 = (size_t)(B*S*H);
  size_t numThreads_1 = (size_t)min(totalThreads_1, 256UL);
  size_t numThreadBlocks_1 = DIVUP(totalThreads_1, numThreads_1);
  binOpFunc0<<<numThreadBlocks_1, numThreads_1, 0, stream>>>(B, S, H, r, S0, S2, comm_size, rank);
  CUDACHECK(cudaEventRecord(stoppipeline_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stoppipeline_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipeline_parallel,stoppipeline_parallel));
  elapsedTimebinOpFunc0 += elapsedTime;


}
int main(int argc, char** argv){
  //Get number of gpus in the node
  int N_GPUs;
  CUDACHECK(cudaGetDeviceCount(&N_GPUs));
  MPI_Init(&argc, &argv);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % N_GPUs));
  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, comm_size, id, rank);
  if (argc < 2) { printf("Specify epochs as command arg"); return 1;}
   int epochs = atoi(argv[1]);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasHandle_t cublasHandle;
  CUBLASCHECK(cublasCreate(&cublasHandle));
  CUBLASCHECK(cublasSetStream(cublasHandle, stream));
  CUBLASCHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
  MPI_Barrier(MPI_COMM_WORLD);

  int array_B[] = {8, 16};
  for (int iter_B = 0; iter_B< sizeof(array_B)/sizeof(array_B[0]);iter_B++) {
    int B = array_B[iter_B];
    size_t S = 1024;
    size_t H = 3072;
    // Inputs
    half* w;
    CUDACHECK(cudaMalloc(&w, DIVUP((H*H), comm_size) * sizeof(half)));
    cudaMemRandInt(w, DIVUP((H*H), comm_size));
    half* b;
    CUDACHECK(cudaMalloc(&b, H * sizeof(half)));
    cudaMemRandInt(b, H);
    half* in;
    CUDACHECK(cudaMalloc(&in, DIVUP((B*S*H), comm_size) * sizeof(half)));
    cudaMemRandInt(in, DIVUP((B*S*H), comm_size));
    half* r;
    CUDACHECK(cudaMalloc(&r, (B*S*H) * sizeof(half)));
    cudaMemRandInt(r, (B*S*H));

    // Outputs
    half* S2;
    CUDACHECK(cudaMalloc(&S2, (B*S*H) * sizeof(half)));

    // Intermediates
    half* S0;
    CUDACHECK(cudaMalloc(&S0, (B*S*H) * sizeof(half)));
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeAllReduce = 0;
    float elapsedTimematMul0 = 0;
    for(int iter = 0; iter < epochs; iter++) {
      pipeline_parallel(B, S, H, w, b, in, r, S2, S0, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimematMul0, comm, stream, comm_size, rank, cublasHandle); 
    }
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(in));
    CUDACHECK(cudaFree(r));
    CUDACHECK(cudaFree(S2));
    CUDACHECK(cudaFree(S0));
    if (rank == 0) 
      printf("{B: %ld, S: %ld, H: %ld, Epochs: %d, binOpFunc0: %f, AllReduce: %f, matMul0: %f, Total: %f}\n", B, S, H, epochs, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimematMul0, elapsedTimebinOpFunc0 + elapsedTimeAllReduce + elapsedTimematMul0);
  }
  MPI_Finalize();
}
