#include "header.h"
#include "cutlass-matmul.h"
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

__global__ void binOpFunc0(int B, int S, int H, half * b, half * S0, half * S2, int comm_size, int rank) {
  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  S2[i0] = (S0[i0] + b[i0%H]);
}

__global__ void binOpFunc1(int B, int S, int H, half * r, half * S0, half * S2, half * S3, int comm_size, int rank) {
  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  curandState curandState0;
  curand_init(0, 0, 0, &curandState0);
  S3[i0] = ((curand_uniform(&curandState0) < 0.5 ? S2[i0] : (half) 0) + r[i0]);
}

void model_parallel(int B, int S, int H, half* w, half* b, half* in, half* r, half* S3, half* S0, half* S2, float& elapsedTimebinOpFunc1, float& elapsedTimebinOpFunc0, float& elapsedTimeAllReduce, float& elapsedTimematMul0, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank, cublasHandle_t cublasHandle){
  cudaEvent_t startmodel_parallel, stopmodel_parallel;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startmodel_parallel));
  CUDACHECK(cudaEventCreate(&stopmodel_parallel));

  CUDACHECK(cudaEventRecord(startmodel_parallel, stream));
  matMul0(B, S, H, w, in, S0, cublasHandle, comm_size, rank);
  CUDACHECK(cudaEventRecord(stopmodel_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stopmodel_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startmodel_parallel,stopmodel_parallel));
  elapsedTimematMul0 += elapsedTime;

  CUDACHECK(cudaEventRecord(startmodel_parallel, stream));
  ncclAllReduce(S0, S0, (B*S*H), ncclHalf,ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stopmodel_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stopmodel_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startmodel_parallel,stopmodel_parallel));
  elapsedTimeAllReduce += elapsedTime;

  CUDACHECK(cudaEventRecord(startmodel_parallel, stream));
  size_t totalThreads_1 = (size_t)(B*S*H);
  size_t numThreads_1 = (size_t)min(totalThreads_1, 256UL);
  size_t numThreadBlocks_1 = DIVUP(totalThreads_1, numThreads_1);
  binOpFunc0<<<numThreadBlocks_1, numThreads_1, 0, stream>>>(B, S, H, b, S0, S2, comm_size, rank);
  CUDACHECK(cudaEventRecord(stopmodel_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stopmodel_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startmodel_parallel,stopmodel_parallel));
  elapsedTimebinOpFunc0 += elapsedTime;

  CUDACHECK(cudaEventRecord(startmodel_parallel, stream));
  size_t totalThreads_2 = (size_t)(B*S*H);
  size_t numThreads_2 = (size_t)min(totalThreads_2, 256UL);
  size_t numThreadBlocks_2 = DIVUP(totalThreads_2, numThreads_2);
  binOpFunc1<<<numThreadBlocks_2, numThreads_2, 0, stream>>>(B, S, H, r, S0, S2, S3, comm_size, rank);
  CUDACHECK(cudaEventRecord(stopmodel_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stopmodel_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startmodel_parallel,stopmodel_parallel));
  elapsedTimebinOpFunc1 += elapsedTime;


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
    half* S3;
    CUDACHECK(cudaMalloc(&S3, (B*S*H) * sizeof(half)));

    // Intermediates
    half* S0;
    half* S2;
    CUDACHECK(cudaMalloc(&S0, (B*S*H) * sizeof(half)));
    CUDACHECK(cudaMalloc(&S2, (B*S*H) * sizeof(half)));
    float elapsedTimebinOpFunc1 = 0;
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeAllReduce = 0;
    float elapsedTimematMul0 = 0;
    for(int iter = 0; iter < epochs; iter++) {
      model_parallel(B, S, H, w, b, in, r, S3, S0, S2, elapsedTimebinOpFunc1, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimematMul0, comm, stream, comm_size, rank, cublasHandle); 
    }
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(in));
    CUDACHECK(cudaFree(r));
    CUDACHECK(cudaFree(S3));
    CUDACHECK(cudaFree(S0));
    CUDACHECK(cudaFree(S2));
    if (rank == 0) 
      printf("{B: %ld, S: %ld, H: %ld, Epochs: %d, binOpFunc1: %f, binOpFunc0: %f, AllReduce: %f, matMul0: %f, Total: %f}\n", B, S, H, epochs, elapsedTimebinOpFunc1, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimematMul0, elapsedTimebinOpFunc1 + elapsedTimebinOpFunc0 + elapsedTimeAllReduce + elapsedTimematMul0);
  }
  MPI_Finalize();
}
