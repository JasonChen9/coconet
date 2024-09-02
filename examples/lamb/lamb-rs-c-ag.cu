#include "header.h"
__global__ void binOpFunc0(int N, float lr, float beta1, float beta2, float gamma, float * w, float * g, float * m, float * v, float * S5, float * S6, float * S7, float * S8, int comm_size, int rank) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    *S7 = 0;
    *S6 = 0;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < DIVUP(N, comm_size); i0 += gridDim.x * blockDim.x) {
     * w = 1;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < DIVUP(N, comm_size); i0 += gridDim.x * blockDim.x) {
    v[i0] = ((v[i0] * beta2) + ((g[DIVUP(N, comm_size) * rank + i0] * (1 - beta2)) * g[DIVUP(N, comm_size) * rank + i0]));
    float S4;
    S4 = (v[i0] / (1 - beta2));
    m[i0] = ((m[i0] * beta1) + (g[DIVUP(N, comm_size) * rank + i0] * (1 - beta1)));
    float S3;
    S3 = (m[i0] / (1 - beta1));
    S5[i0] = ((S3 / (sqrt(S4))) + (w[i0] * gamma));
     * S5 = 1;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < DIVUP(N, comm_size); i0 += gridDim.x * blockDim.x) {
    S8[i0] = (w[i0] - (((S7[0] / S6[0]) * lr) * S5[i0]));
  }
}

void lamb(int N, float lr, float beta1, float beta2, float epsilon, float gamma, float* w, float* g, float* m, float* v, float* S8, float* S5, float* S6, float* S7, float& elapsedTimeAllGather, float& elapsedTimebinOpFunc0, float& elapsedTimeReduceScatter, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startlamb, stoplamb;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startlamb));
  CUDACHECK(cudaEventCreate(&stoplamb));

  CUDACHECK(cudaEventRecord(startlamb, stream));
  ncclReduceScatter(g, g, DIVUP(N, comm_size), ncclFloat32, ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stoplamb, stream));
  CUDACHECK(cudaEventSynchronize(stoplamb));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startlamb,stoplamb));
  elapsedTimeReduceScatter += elapsedTime;

  CUDACHECK(cudaEventRecord(startlamb, stream));
  dim3 numThreads_0 = {256, 1, 1};
  dim3 numThreadBlocks_0 = {1, 1, 1};
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&numThreadBlocks_0.x, (void*)binOpFunc0, numThreads_0.x, 0));
  numThreadBlocks_0.x = 80 * numThreadBlocks_0.x;
  void* args_0[] = {&N, &lr, &beta1, &beta2, &gamma, &w, &g, &m, &v, &S5, &S6, &S7, &S8, &comm_size, &rank};
  CUDACHECK(cudaLaunchCooperativeKernel((void*)binOpFunc0, numThreadBlocks_0, numThreads_0, args_0, 0, stream));
;
  CUDACHECK(cudaEventRecord(stoplamb, stream));
  CUDACHECK(cudaEventSynchronize(stoplamb));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startlamb,stoplamb));
  elapsedTimebinOpFunc0 += elapsedTime;

  CUDACHECK(cudaEventRecord(startlamb, stream));
  ncclAllGather(S8, w, DIVUP(N, comm_size), ncclFloat32, comm, stream);
  CUDACHECK(cudaEventRecord(stoplamb, stream));
  CUDACHECK(cudaEventSynchronize(stoplamb));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startlamb,stoplamb));
  elapsedTimeAllGather += elapsedTime;


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
  MPI_Barrier(MPI_COMM_WORLD);

  for (int __i = 10; __i < 30; __i++) {
    size_t N = 1 << __i;
    // Inputs
    float* g;
    CUDACHECK(cudaMalloc(&g, N * sizeof(float)));
    cudaMemRandInt(g, N);
    float* w;
    CUDACHECK(cudaMalloc(&w, N * sizeof(float)));
    cudaMemRandInt(w, N);
    float* m;
    CUDACHECK(cudaMalloc(&m, DIVUP(N, comm_size) * sizeof(float)));
    cudaMemRandInt(m, DIVUP(N, comm_size));
    float* v;
    CUDACHECK(cudaMalloc(&v, DIVUP(N, comm_size) * sizeof(float)));
    cudaMemRandInt(v, DIVUP(N, comm_size));
    float lr;
    lr = 1.0f;
    float beta1;
    beta1 = 1.0f;
    float beta2;
    beta2 = 1.0f;
    float epsilon;
    epsilon = 1.0f;
    float gamma;
    gamma = 1.0f;

    // Outputs

    // Intermediates
    float* S8;
    float* S5;
    float* S6;
    float* S7;
    CUDACHECK(cudaMalloc(&S8, DIVUP(N, comm_size) * sizeof(float)));
    CUDACHECK(cudaMalloc(&S5, DIVUP(N, comm_size) * sizeof(float)));
    CUDACHECK(cudaMalloc(&S6, DIVUP(1, comm_size) * sizeof(float)));
    CUDACHECK(cudaMalloc(&S7, 1 * sizeof(float)));
    float elapsedTimeAllGather = 0;
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeReduceScatter = 0;
    for(int iter = 0; iter < epochs; iter++) {
      lamb(N, lr, beta1, beta2, epsilon, gamma, w, g, m, v, S8, S5, S6, S7, elapsedTimeAllGather, elapsedTimebinOpFunc0, elapsedTimeReduceScatter, comm, stream, comm_size, rank); 
    }
    CUDACHECK(cudaFree(g));
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(m));
    CUDACHECK(cudaFree(v));
    CUDACHECK(cudaFree(S8));
    CUDACHECK(cudaFree(S5));
    CUDACHECK(cudaFree(S6));
    CUDACHECK(cudaFree(S7));
    if (rank == 0) 
      printf("{SZ: %ld, Epochs: %d, AllGather: %f, binOpFunc0: %f, ReduceScatter: %f, Total: %f}\n", N, epochs, elapsedTimeAllGather, elapsedTimebinOpFunc0, elapsedTimeReduceScatter, elapsedTimeAllGather + elapsedTimebinOpFunc0 + elapsedTimeReduceScatter);
  }
  MPI_Finalize();
}
