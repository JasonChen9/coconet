#include "header.h"
void pipeline_parallel(int B, int S, int H, half* w, half* b, half* in, half* r, half* S0, half* S3, float& elapsedTimeFusedAllReduce, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank){
  cudaEvent_t startpipeline_parallel, stoppipeline_parallel;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startpipeline_parallel));
  CUDACHECK(cudaEventCreate(&stoppipeline_parallel));

  CUDACHECK(cudaEventRecord(startpipeline_parallel, stream));
  NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, (half*)g, w, (half*)w, m, v, N, ncclHalf, comm, ncclSum, stream));
  CUDACHECK(cudaEventRecord(stoppipeline_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stoppipeline_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipeline_parallel,stoppipeline_parallel));
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

  int array_B[] = {8, 16};
  for (int iter_B = 0; iter_B< sizeof(array_B)/sizeof(array_B[0]);iter_B++) {
    int B = array_B[iter_B];
    size_t S = 1024;
    size_t H = 3072;
    // Inputs
    half* w;
    CUDACHECK(cudaMalloc(&w, DIVUP((H*H), comm_size)*comm_size * sizeof(half)));
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
    float elapsedTimeFusedAllReduce = 0;
    for(int iter = 0; iter < epochs; iter++) {
      pipeline_parallel(B, S, H, w, b, in, r, S0, S3, elapsedTimeFusedAllReduce, comm, stream, comm_size, rank); 
    }
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(in));
    CUDACHECK(cudaFree(r));
    CUDACHECK(cudaFree(S3));
    if (rank == 0) 
      printf("{B: %ld, S: %ld, H: %ld, Epochs: %d, FusedAllReduce: %f, Total: %f}\n", B, S, H, epochs, elapsedTimeFusedAllReduce, elapsedTimeFusedAllReduce);
  }
  MPI_Finalize();
}
