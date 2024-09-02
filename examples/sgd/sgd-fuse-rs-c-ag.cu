#include "header.h"
void sgd(int N, float lr, float* g, float* w, float& elapsedTimeFusedAllReduce, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startsgd, stopsgd;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startsgd));
  CUDACHECK(cudaEventCreate(&stopsgd));

  CUDACHECK(cudaEventRecord(startsgd, stream));
  NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, (half*)g, w, (half*)w, m, v, N, ncclHalf, comm, ncclSum, stream));
  CUDACHECK(cudaEventRecord(stopsgd, stream));
  CUDACHECK(cudaEventSynchronize(stopsgd));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startsgd,stopsgd));
  elapsedTimeFusedAllReduce += elapsedTime;


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
    float elapsedTimeFusedAllReduce = 0;
    for(int iter = 0; iter < epochs; iter++) {
      sgd(N, lr, g, w, elapsedTimeFusedAllReduce, comm, stream, comm_size, rank); 
    }
    CUDACHECK(cudaFree(g));
    CUDACHECK(cudaFree(w));
    if (rank == 0) 
      printf("{SZ: %ld, Epochs: %d, FusedAllReduce: %f, Total: %f}\n", N, epochs, elapsedTimeFusedAllReduce, elapsedTimeFusedAllReduce);
  }
  MPI_Finalize();
}
