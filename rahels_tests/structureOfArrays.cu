/*
File: tiled_solution.cu
Author: Celyn Jacobs
Class: ECE569
Description: Contains tiled multiplication kernel and associated launch code.
*/


#include <wb.h>
#include <cuda_fp16.h>

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
#define no_r_b 1024 //size of shared memory array
#define FULL_MASK 0xffffffff
#define DEBUG_THREAD 2000
#define DEBUG_BLOCK 20
#define NUM_ELEMS_FOR_TESTING 1000000

__constant__ int RAND[no_r_b];

/*
struct mf_node{
	int u,v;
	float r;
};
*/

struct mf_node{
    int u[no_r_b];
	int v[no_r_b];
	float r[no_r_b];
};

mf_node* readInputAsMF(float* input, int numRows, int numCols){
  mf_node* inputMF = (mf_node *) malloc(numRows * numCols * sizeof(mf_node));
  for(int u = 0; u < numRows; u++){
    for(int v = 0; v < numCols; v++){
      inputMF.u[u*numCols+v] = u; // row index
      inputMF.v[u*numCols+v] = v; // column index
      inputMF.r[u*numCols+v] = input[u*numCols+v]; // rating value
    }
  }
  return inputMF;
}

float getRMSE(mf_node *R, half *P, half* Q, int numRRows, int numRColumns){
  // Q IS NOT A COLUMN MATRIX. IT'S A ROW MATRIX JUST LIKE P
  float rmse = 0.0;
  int numElems = numRRows * numRColumns;
  float sum, tmp, actual;

  for(int i = 0; i < numRRows; ++i){ // i is the row in R & P
    for(int j = 0; j < numRColumns; ++j){ // j is the column in R & Q
      sum = 0.0;
      for(int k = 0; k < K; ++k){
        sum += __half2float(P[i*K + k]) * __half2float(Q[j*K + k]);
      }
      //actual = R[i*numRColumns + j].r;
      actual = R.r[i*numRColumns + j];
     // printf("Getting error between %f and %f\n", sum, actual);
      
      rmse += (actual-sum) * (actual-sum);
    }
  }

  rmse = rmse / numElems;
  return rmse;
}



__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
                            mf_node *R,
                            long long numElems,
                            half *p,
                            half *q,
                            int k, //feature dimension vector 
                            int num_iters,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda,
                            float beta,
                            float initialLearningRate
                            )
{

/*
In MF, one SGD update consists of four steps: 
1) read one sample (r[u,v] ) from the rating matrix, 
2) read two feature vectors (pu , qv ), 
3) compute prediction error(r[u,v] − pu * qv ), and 
4) update the features. Except for the first step, other three steps are all vector operations at length k.
*/

// OPTIMIZATION TO ACCESS GLOBAL MEMORY IN COALESCED FASHION BUT LOAD INTO SHARED MEM RANDOMLY
    __shared__ mf_node sh_rating[no_r_b];

   
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int index = blockId * blockDim.x + threadIdx.x;

    if (index < blockDim.x * blockDim.y){ //if < 1024, we're in the 0th plane
    //if(index < numElems) {
    int rand_idx = RAND[threadIdx.x];
    if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\n Got random index %d \n\n", rand_idx);}
    /*
    sh_rating[rand_idx].u = __ldg(&R[index].u);
    sh_rating[rand_idx].v = __ldg(&R[index].v);
    sh_rating[rand_idx].r = __ldg(&R[index].r); 
    */
    sh_rating.u[rand_idx] = __ldg(&R.u[index]);
    sh_rating.v[rand_idx] = __ldg(&R.v[index]);
    sh_rating.r[rand_idx] = __ldg(&R.r[index]); 

    __syncthreads();

    // OPTIMIZATION: PULLED THIS OUT OF THE MAIN LOOP BECAUSE REDUNDANT
    int lane_id = threadIdx.x % 32;
    int local_wid = threadIdx.x / 32;
    // int wid = 4*blockIdx.x + local_wid;

    //////////////////
    // OPTIMIZATION to accomodate shared mem indexing
    int rat_per_block = blockDim.x / 32;
    /////////////////

    //persistant thread
    for(int ite = 0; ite < num_iters; ite++){ // repeat running thread to improve error rate
        /*
        __ldg : Read-Only Data Cache Load Function
        T __ldg(const T* address);
        returns the data of type T located at address address,
        */
        // float tmp_lrate = __ldg(&dynamic_rate[ite]);
        float tmp_lrate = initialLearningRate/(1+beta * powf(ite,1.5)); // decreases learning rate every iteration

        // update_count_this_block is st in egypt paper
        for(int update_ite = 0; update_ite < update_count_this_block; update_ite++){
            /* OPTIMIZATION WITH SHARED MEM MEANS WE DONT NEED TO CALCULATE RAND INDEX EVERY TIME
            long long start_id = 0;
            if(lane_id == 0){ 
                long long origin = (long long)(curand_uniform(&state[wid])*nnz);  
                start_id = origin%nnz;
                //start_id == 0;
            }
            start_id = __shfl(start_id, 0); 
            */
            
            for(int i = 0; i < update_vector_size; i++) // 32 in Egypt paper
            {
                ///////////////////
                // OPTIMIZATION TO ACCESS SHARED MEM INSTEAD OF GLOBAL
                //every thread in the warp works together to compute R. 
                // Each thread does 4 dot product elements times 32 threads for a total of k=128
                int temp = local_wid + i * rat_per_block; // every thread in a warp indexes to the same element and that element is in the same lane
                /*
                int u = sh_rating[temp].u;
                int v = sh_rating[temp].v;
                float r = sh_rating[temp].r;
                */
               
                int u = sh_rating.v[temp];
                int v = sh_rating.v[temp];
                float r = sh_rating.r[temp];
                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){
                  printf("\nHandling rating at index %d in shared mem corresponding to rating in row %d, column %d, with value %f.\n\n", temp, u, v, r);
                }

                //read the p & q into register file.
                int base_p = u*k;
                int base_q = v*k;

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\n base_p is %d and base_q is %d \n\n", base_p,base_q);}

                float tmp_p1 = __half2float(p[base_p + lane_id + 0]);
                float tmp_q1 = __half2float(q[base_q + lane_id + 0]);
            
                float tmp_p2 = __half2float(p[base_p + lane_id + 32]);
                float tmp_q2 = __half2float(q[base_q + lane_id + 32]);
            
                float tmp_p3 = __half2float(p[base_p + lane_id + 64]);
                float tmp_q3 = __half2float(q[base_q + lane_id + 64]);
            
                float tmp_p4 = __half2float(p[base_p + lane_id + 96]);
                float tmp_q4 = __half2float(q[base_q + lane_id + 96]);

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\n temp p is %.2f,%.2f,%.2f,%.2f and temp q is %.2f,%.2f,%.2f,%.2f \n\n", 
                tmp_p1,tmp_p2, tmp_p3, tmp_p4, tmp_q1, tmp_q2, tmp_q3, tmp_q4);}

                float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

                //get dot product.
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 16);
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 8);
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 4);
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 2);
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 1); // at this point thread 0 has the total sum

                tmp_product = __shfl_sync(FULL_MASK,tmp_product,0); // broadcasts the value in thread 0 to all the other threads

                float ruv = r - tmp_product; //get error

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nerror = %f\n\n", ruv);}

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nwriteback idx = %d\n\n", base_p+lane_id);}

                //update
                //only works for k=blockDim.x=128
                p[base_p + lane_id +  0] = __float2half(tmp_p1 + tmp_lrate*(ruv*tmp_q1 - lambda*tmp_p1));
                q[base_q + lane_id +  0] = __float2half(tmp_q1 + tmp_lrate*(ruv*tmp_p1 - lambda*tmp_q1));

                p[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lrate*(ruv*tmp_q2 - lambda*tmp_p2));
                q[base_q + lane_id + 32] = __float2half(tmp_q2 + tmp_lrate*(ruv*tmp_p2 - lambda*tmp_q2));

                p[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lrate*(ruv*tmp_q3 - lambda*tmp_p3));
                q[base_q + lane_id + 64] = __float2half(tmp_q3 + tmp_lrate*(ruv*tmp_p3 - lambda*tmp_q3));

                p[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lrate*(ruv*tmp_q4 - lambda*tmp_p4));
                q[base_q + lane_id + 96] = __float2half(tmp_q4 + tmp_lrate*(ruv*tmp_p4 - lambda*tmp_q4));

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nwriteback idx = %d\n\n", base_p+lane_id+96);}

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){
                  printf("\nwroteback to p %f,%f,%f,%f at index %d and wroteback to q %f,%f,%f,%f at index %d \n\n", 
                  p[base_p + lane_id +  0], p[base_p + lane_id +  32], p[base_p + lane_id +  64], p[base_p + lane_id +  96], base_p + lane_id + 96,
                  q[base_p + lane_id +  0], q[base_p + lane_id +  32], q[base_p + lane_id +  64], q[base_p + lane_id +  96], base_q + lane_id + 96);
                }
            }    
        }
    }

    __syncthreads();
    if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nThe value in index 0 is %f\n", p[0]);}
    }//if index < BlockDim.x * blockDim.y
   // }   //if index < numElems 
}

mf_node* readInputAsMF(float* input, int numRows, int numCols){
  mf_node* inputMF = (mf_node *) malloc(numRows * numCols * sizeof(mf_node));

  for(int u = 0; u < numRows; u++){
    for(int v = 0; v < numCols; v++){
      inputMF[u*numCols+v].u = u; // row index
      inputMF[u*numCols+v].v = v; // column index
      inputMF[u*numCols+v].r = input[u*numCols+v]; // rating value
    }
  }

  return inputMF;
}

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

// for ease of testing, A = R, B = P, C = Q
int main(int argc, char **argv) {
  wbArg_t args;
  half *hostP; // The A matrix
  half *hostQ; // The B matrix
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
  int numElems;
  int* hostRandGlobal;
  int* deviceRandGlobal;

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  temp = (float *)wbImport(wbArg_getInputFile(args, 0), &numRRows,
                            &numRColumns);

  printf("Matrix has %d rows and %d columns\n\n", numRRows,numRColumns);
                
  hostR = readInputAsMF(temp,numRRows,numRColumns);

  int indx = 1001;
  printf("\nElement %d at row %d and column %d is %f\n\n", indx, hostR[indx].u,hostR[indx].v,hostR[indx].r);

  numElems = numRRows * numRColumns;

  // hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
  //                          &numBColumns);
                            
  //@@ Set numCRows and numCColumns
  numPRows    =  numRRows;   // set to correct value
  numPColumns = K;   // set to correct value
  numQRows = K; // THIS IS WRONG BECAUSE THEY DO Q IN A WEIRD WAY
  numQColumns = numRColumns;
  //@@ Allocate the hostC matrix
  
  hostP = (half *)malloc((numPRows*numPColumns) * sizeof(half));
  hostQ = (half *)malloc((numQRows*numQColumns) * sizeof(half));

  // have to set to nonzero initially or the algorithm won't do anything
  for(int i = 0; i < numPRows * numPColumns; ++i)
    hostP[i] = 0.01;
  for(int i = 0; i < numQRows * numQColumns; ++i)
    hostQ[i] = 0.01;


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
  cudaMemcpy(deviceP,hostP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceQ,hostQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13.
  // Here, we perform a de facto ceil() operation on integers using integer arithmetic
  int st = 1;

  dim3 mygrid(((numRRows * numRColumns)-1)/no_r_b + 1);
  dim3 myblock(no_r_b/st);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  float lambda = 0.05;//.05
  float beta = 0.3;
  float initialLearningRate = 0.01;//0.08 og
  float update_vector_size = 32;
  int num_iters = 10;
  float rmse;

  
  cudaEventRecord(start, 0);

  for (int i = 0; i < no_r_b; i++) // initialize const array of random indices
    hostRand[i] = i;
  shuffle(hostRand,no_r_b);

  cudaMemcpyToSymbol(RAND,hostRand, no_r_b * sizeof(int));
  sgd_k128_kernel_hogwild_warp32_lrate<<<mygrid,myblock>>>(
                            deviceR,
                            numElems,
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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start, stop);
    printf("\n");
    printf("Total compute time (ms) %f for our kernel\n",elapsedTime);
    printf("\n");

    // Now get rmse for our kernel
    cudaMemcpy(hostP,deviceP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostQ,deviceQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyDeviceToHost);
    rmse = getRMSE(hostR, hostP, hostQ, numRRows, numRColumns); // in future can make this GPU kernel but for now we do serial
    printf("\n");
    printf("RMSE for our kernel: %f \n",rmse);
    printf("\n");
    
  
  cudaEventRecord(start, 0);

  hostRandGlobal = (int* )malloc(numElems*sizeof(int));
  for (int i = 0; i < numElems; i++)
    hostRandGlobal[i] = i;
  shuffle(hostRandGlobal,numElems);

  cudaMalloc((void **) &deviceRandGlobal, numElems*sizeof(int));
  cudaMemcpy(deviceRandGlobal,hostRandGlobal,numElems*sizeof(int),cudaMemcpyHostToDevice);

  theirKernel<<<mygrid,myblock>>>(
                            deviceR,
                            numElems,
                            deviceP,
                            deviceQ,
                            K, 
                            num_iters,
                            st,
                            update_vector_size,
                            lambda,
                            beta,
                            initialLearningRate,
                            deviceRandGlobal
                            );
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start, stop);
    printf("\n");
    printf("Total compute time (ms) %f for their kernel\n",elapsedTime);
    printf("\n");

    // Now get rmse for their kernel
    cudaMemcpy(hostP,deviceP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostQ,deviceQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyDeviceToHost);
    rmse = getRMSE(hostR, hostP, hostQ, numRRows, numRColumns); // in future can make this GPU kernel but for now we do serial
    printf("\n");
    printf("RMSE for their kernel: %f \n",rmse);
    printf("\n");
  

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostP,deviceP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyDeviceToHost);
  cudaMemcpy(hostQ,deviceQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

/*
  printf("\n\n");
  for(int i = 0; i < numPRows; ++i){
    for(int j = 0; j < numPColumns; ++j){
      printf("%f ", __half2float(hostP[i * numPColumns + j]));
    }
    printf("\n");
  }
  */

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
  //free(hostRandGlobal);

  return 0;
}