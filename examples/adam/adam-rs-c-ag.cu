#include "header.h"
__global__ void binOpFunc0(int N, float beta1, float beta2, float * g, float * w, float * m, float * v, float * S5, int comm_size, int rank) {
  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  v[i0] = ((v[i0] * beta2) + ((g[DIVUP(N, comm_size) * rank + i0] * (1 - beta2)) * g[DIVUP(N, comm_size) * rank + i0]));
  float S4;
  S4 = (v[i0] / beta2);
  m[i0] = ((m[i0] * beta1) + (g[DIVUP(N, comm_size) * rank + i0] * (1 - beta1)));
  float S3;
  S3 = (m[i0] / beta1);
  S5[i0] = (w[DIVUP(N, comm_size) * rank + i0] - (S3 / S4));
}

void adam(int N, float lr, float beta1, float beta2, float* g, float* w, float* m, float* v, float* S5, float& elapsedTimebinOpFunc0, float& elapsedTimeAllGather, float& elapsedTimeReduceScatter, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startadam, stopadam;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startadam));
  CUDACHECK(cudaEventCreate(&stopadam));

  CUDACHECK(cudaEventRecord(startadam, stream));
  ncclReduceScatter(g, g, DIVUP(N, comm_size), ncclFloat32, ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stopadam, stream));
  CUDACHECK(cudaEventSynchronize(stopadam));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startadam,stopadam));
  elapsedTimeReduceScatter += elapsedTime;

  CUDACHECK(cudaEventRecord(startadam, stream));
  size_t totalThreads_0 = (size_t)DIVUP(N, comm_size);
  size_t numThreads_0 = (size_t)min(totalThreads_0, 256UL);
  size_t numThreadBlocks_0 = DIVUP(totalThreads_0, numThreads_0);
  binOpFunc0<<<numThreadBlocks_0, numThreads_0, 0, stream>>>(N, beta1, beta2, g, w, m, v, S5, comm_size, rank);
  CUDACHECK(cudaEventRecord(stopadam, stream));
  CUDACHECK(cudaEventSynchronize(stopadam));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startadam,stopadam));
  elapsedTimebinOpFunc0 += elapsedTime;

  CUDACHECK(cudaEventRecord(startadam, stream));
  ncclAllGather(S5, w, DIVUP(N, comm_size), ncclFloat32, comm, stream);
  CUDACHECK(cudaEventRecord(stopadam, stream));
  CUDACHECK(cudaEventSynchronize(stopadam));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startadam,stopadam));
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

    // Outputs

    // Intermediates
    float* S5;
    CUDACHECK(cudaMalloc(&S5, DIVUP(N, comm_size) * sizeof(float)));
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeAllGather = 0;
    float elapsedTimeReduceScatter = 0;
    for(int iter = 0; iter < epochs; iter++) {
      adam(N, lr, beta1, beta2, g, w, m, v, S5, elapsedTimebinOpFunc0, elapsedTimeAllGather, elapsedTimeReduceScatter, comm, stream, comm_size, rank); 
    }
    CUDACHECK(cudaFree(g));
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(m));
    CUDACHECK(cudaFree(v));
    CUDACHECK(cudaFree(S5));
    if (rank == 0) 
      printf("{SZ: %ld, Epochs: %d, binOpFunc0: %f, AllGather: %f, ReduceScatter: %f, Total: %f}\n", N, epochs, elapsedTimebinOpFunc0, elapsedTimeAllGather, elapsedTimeReduceScatter, elapsedTimebinOpFunc0 + elapsedTimeAllGather + elapsedTimeReduceScatter);
  }
  MPI_Finalize();
}
