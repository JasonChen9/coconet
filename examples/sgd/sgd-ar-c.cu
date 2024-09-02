#include "header.h"
__global__ void binOpFunc0(int N, float lr, float * g, float * w, int comm_size, int rank) {
  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  w[i0] = (w[i0] - (g[i0] * lr));
}

void sgd(int N, float lr, float* g, float* w, float& elapsedTimebinOpFunc0, float& elapsedTimeAllReduce, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startsgd, stopsgd;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startsgd));
  CUDACHECK(cudaEventCreate(&stopsgd));

  CUDACHECK(cudaEventRecord(startsgd, stream));
  ncclAllReduce(g, g, N, ncclFloat32,ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stopsgd, stream));
  CUDACHECK(cudaEventSynchronize(stopsgd));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startsgd,stopsgd));
  elapsedTimeAllReduce += elapsedTime;

  CUDACHECK(cudaEventRecord(startsgd, stream));
  size_t totalThreads_0 = (size_t)N;
  size_t numThreads_0 = (size_t)min(totalThreads_0, 256UL);
  size_t numThreadBlocks_0 = DIVUP(totalThreads_0, numThreads_0);
  binOpFunc0<<<numThreadBlocks_0, numThreads_0, 0, stream>>>(N, lr, g, w, comm_size, rank);
  CUDACHECK(cudaEventRecord(stopsgd, stream));
  CUDACHECK(cudaEventSynchronize(stopsgd));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startsgd,stopsgd));
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
    float lr;
    lr = 1.0f;

    // Outputs
    float elapsedTimebinOpFunc0 = 0;
    float elapsedTimeAllReduce = 0;
    for(int iter = 0; iter < epochs; iter++) {
      sgd(N, lr, g, w, elapsedTimebinOpFunc0, elapsedTimeAllReduce, comm, stream, comm_size, rank); 
    }
    CUDACHECK(cudaFree(g));
    CUDACHECK(cudaFree(w));
    if (rank == 0) 
      printf("{SZ: %ld, Epochs: %d, binOpFunc0: %f, AllReduce: %f, Total: %f}\n", N, epochs, elapsedTimebinOpFunc0, elapsedTimeAllReduce, elapsedTimebinOpFunc0 + elapsedTimeAllReduce);
  }
  MPI_Finalize();
}
