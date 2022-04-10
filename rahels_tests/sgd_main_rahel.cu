
#include <wb.h>
#include <cuda_fp16.h>
#include "sgd_rahel.h"
#include "sgd_datatypes.h"
#include "sgd_rahel.cu"
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define K 128
#define no_r_b 1024
#define FULL_MASK 0xffffffff


// for ease of testing, A = R, B = P, C = Q
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostP; // The A matrix
  float *hostQ; // The B matrix
  mf_node *hostR; // The output C matrix
  float *temp;
  half *deviceP; // A matrix on device
  half *deviceQ; // B matrix on device
  mf_node *deviceR; // C matrix on device
  int numPRows;    // number of rows in the matrix A
  int numPColumns; // number of columns in the matrix A
  int numQRows;    // number of rows in the matrix B
  int numQColumns; // number of columns in the matrix B
  int numRRows;    // number of rows in the matrix C(you have to set this)
  int numRColumns; // number of columns in the matrix C (you have to set
                   // this)
  int hostRand[no_r_b];

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  temp = (float *)wbImport(wbArg_getInputFile(args, 0), &numRRows,
                            &numRColumns);
  hostR = readInputAsMF(temp,numRRows,numRColumns);
	//random access for r.
  for (int i = 0; i < no_r_b; i++) // initialize const array of random indices
    hostRand[i] = rand() % no_r_b;

  // hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
  //                          &numBColumns);
                            
  //@@ Set numCRows and numCColumns
  numPRows    =  numRRows;    
  numPColumns = K;   
  numQRows = K;
  numQColumns = numRColumns;
  //@@ Allocate the hostC matrix
  
  hostP = (float *)malloc((numPRows*numPColumns) * sizeof(half));
  hostQ = (float *)malloc((numQRows*numQColumns) * sizeof(half));

  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceP, (numPRows*numPColumns)*sizeof(half));
  cudaMalloc((void **) &deviceQ, (numQRows*numQColumns)*sizeof(half));
  cudaMalloc((void **) &deviceR, (numRRows*numRColumns)*sizeof(mf_node));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceR,hostR,(numRRows*numRColumns)*sizeof(mf_node),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(RAND,hostRand, no_r_b * sizeof(int));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13.
  // Here, we perform a de facto ceil() operation on integers using integer arithmetic
  int st = 2;

  dim3 mygrid(((numRRows * numRColumns)-1)/no_r_b + 1);
  dim3 myblock(no_r_b/st);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  float lambda = 0.05;
  float beta = 0.3;
  float initialLearningRate = 0.08;
  float update_vector_size = 32;
  int num_iters = 10;

  sgd_k128_kernel_hogwild_warp32_lrate<<<mygrid,myblock>>>(
                            deviceR,
                            deviceP,
                            deviceQ,
                            K, 
                            num_iters,
                            st,
                            update_vector_size,
                            lambda,
                            beta,
                            initialLearningRate
                            );

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostP,deviceP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyDeviceToHost);
  cudaMemcpy(hostQ,deviceQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceP);
  cudaFree(deviceQ);
  cudaFree(deviceR);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostP, numPRows, numPColumns);

  free(hostP);
  free(hostQ);
  free(hostR);
  free(temp);

  return 0;
}
