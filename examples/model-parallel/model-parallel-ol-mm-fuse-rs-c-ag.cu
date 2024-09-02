#include "header.h"
#include "cutlass-matmul.h"
void model_parallel(int B, int S, int H, half* w, half* b, half* in, half* r, half* S3, half* S0, float& elapsedTimeoverlap, ncclComm_t comm, cudaStream_t stream, int comm_size, int rank, Gemm& gemm_op, int iter, int* tileStatusMap, cudaStream_t cutlassStream){
  cudaEvent_t startmodel_parallel, stopmodel_parallel;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&startmodel_parallel));
  CUDACHECK(cudaEventCreate(&stopmodel_parallel));

  CUDACHECK(cudaEventRecord(startmodel_parallel, stream));
  NCCLCHECK(ncclAllReduceOverlapMatMul(in, w, S0, tileStatusMap, B*S*H, B*S, H, DIVUP(H, comm_size), 512, iter, ncclHalf, ncclSum, comm, stream));

  CUTLASS_CHECK(gemm_op(iter, cutlassStream));
  CUDACHECK(cudaEventRecord(stopmodel_parallel, stream));
  CUDACHECK(cudaEventSynchronize(stopmodel_parallel));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startmodel_parallel,stopmodel_parallel));
  elapsedTimeoverlap += elapsedTime;


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
  cudaStream_t cutlassStream;
  cudaStreamCreate(&cutlassStream);
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
    CUDACHECK(cudaMalloc(&S0, (B*S*H) * sizeof(half)));
    float elapsedTimeoverlap = 0;
    Gemm gemm_op;
    int* threadBlockToTileMap;int* tileIdx;int* tileStatusMap;int* chunksForTile;
    getCutlassGemm(&comm, gemm_op, B*S, H, DIVUP((H), comm_size), in, w, S0, threadBlockToTileMap, tileIdx, tileStatusMap, chunksForTile, comm_size, rank);

    for(int iter = 0; iter < epochs; iter++) {
      model_parallel(B, S, H, w, b, in, r, S3, S0, elapsedTimeoverlap, comm, stream, comm_size, rank, gemm_op, iter, tileStatusMap, cutlassStream); 
    }
    CUDACHECK(cudaFree(w));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(in));
    CUDACHECK(cudaFree(r));
    CUDACHECK(cudaFree(S3));
    CUDACHECK(cudaFree(S0));
    if (rank == 0) 
      printf("{B: %ld, S: %ld, H: %ld, Epochs: %d, overlap: %f, Total: %f}\n", B, S, H, epochs, elapsedTimeoverlap, elapsedTimeoverlap);
  }
  MPI_Finalize();
}
