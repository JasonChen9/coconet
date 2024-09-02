#include "header.h"
__global__ void binOpFunc0(int N, float lr, float beta1, float beta2, float gamma, float * w, float * g, float * m, float * v, float * S5, float * S6, float * S7, float * S8, int comm_size, int rank) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    *S7 = 0;
    *S6 = 0;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < N; i0 += gridDim.x * blockDim.x) {
     * w = 1;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < N; i0 += gridDim.x * blockDim.x) {
    v[i0] = ((v[i0] * beta2) + ((g[i0] * (1 - beta2)) * g[i0]));
    float S4;
    S4 = (v[i0] / (1 - beta2));
    m[i0] = ((m[i0] * beta1) + (g[i0] * (1 - beta1)));
    float S3;
    S3 = (m[i0] / (1 - beta1));
    S5[i0] = ((S3 / (sqrt(S4))) + (w[i0] * gamma));
     * S5 = 1;
  }
  this_grid().sync();
  for (int i0 = threadIdx.x + blockDim.x*blockIdx.x; i0 < N; i0 += gridDim.x * blockDim.x) {
    S8[i0] = (w[i0] - (((S7[0] / S6[0]) * lr) * S5[i0]));
  }
}

void lamb(int N, float lr, float beta1, float beta2, float epsilon, float gamma, float* w, float* g, float* m, float* v, float* S8, float* S5, float* S6, float* S7, float& elapsedTimebinOpFunc0, float& elapsedTimeAllReduce, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startlamb, stoplamb;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startlamb));
  CUDACHECK(cudaEventCreate(&stoplamb));

  CUDACHECK(cudaEventRecord(startlamb, stream));
  ncclAllReduce(g, g, N, ncclFloat32,ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stoplamb, stream));
  CUDACHECK(cudaEventSynchronize(stoplamb));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startlamb,stoplamb));
  elapsedTimeAllReduce += elapsedTime;

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
    CUDACHECK(cudaMalloc(&m, N * sizeof(float)));
    cudaMemRandInt(m, N);
    float* v;
    CUDACHECK(cudaMalloc(&v, N * sizeof(float)));
    cudaMemRandInt(v, N);
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
    float* S8;
    CUDACHECK(cudaMalloc(&S8, N * sizeof(float)));

    // Intermediates
    float* S5;
    float* S6;
    float* S7;
    CUDACHECK(cudaMalloc(&S5, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&S6, 1 * sizeof(float)));
    CUDACHECK(cudaMalloc(&S7, 1 * sizeof(float)));
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeAllReduce = 0;
    for(int iter = 0; iter < epochs; iter++) {
      lamb(N, lr, beta1, beta2, epsilon, gamma, w, g, m, v, S8, S5, S6, S7, elapsedTimebinOpFunc0, elapsedTimeAllReduce, comm, stream, comm_size, rank); 
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
      printf("{SZ: %ld, Epochs: %d, binOpFunc0: %f, AllReduce: %f, Total: %f}\n", N, epochs, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimebinOpFunc0 + elapsedTimeAllReduce);
  }
  MPI_Finalize();
}
