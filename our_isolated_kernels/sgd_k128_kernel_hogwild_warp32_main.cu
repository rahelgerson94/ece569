/*
File: tiled_solution.cu
Author: Celyn Jacobs, Rahel Gerson
Class: ECE569
Description: Contains tiled multiplication kernel and associated launch code.
*/


//#include <wb.h>
#include <cuda_fp16.h>
#include <time.h>
/*
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
*/


#define K 128
#define no_r_b 1024
#define FULL_MASK 0xffffffff
#define DEBUG_THREAD 511
#define DEBUG_BLOCK 5
__constant__ int RAND[no_r_b];

struct mf_node
{
	int u,v;
	float r;
};

__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
                            mf_node *R,
//                          long long nnz,
                            half *p,
                            half *q,
//                          curandState *state,
//                            float *dynamic_rate,
                            int k, //feature dimension vector 
                            int num_iters,
//                            int current_iter,
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

//////////////////////////
// OPTIMIZATION TO ACCESS GLOBAL MEMORY IN COALESCED FASHION BUT LOAD INTO SHARED MEM RANDOMLY
    __shared__ mf_node sh_rating[no_r_b];

    for (int i = 0; i < update_count_this_block; ++i){
        int rand_idx = RAND[blockDim.x * i + threadIdx.x];
        sh_rating[rand_idx].u = __ldg(&R[1/2*i+threadIdx.x].u);
        sh_rating[rand_idx].v = __ldg(&R[1/2*i+threadIdx.x].v);
        sh_rating[rand_idx].r = __ldg(&R[1/2*i+threadIdx.x].r);
    }
/////////////////////////

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
                int u = sh_rating[local_wid + i * rat_per_block].u;
                int v = sh_rating[local_wid + i * rat_per_block].v;
                float r = sh_rating[local_wid + i * rat_per_block].r;
                ///////////////////
                
                /*
                int offset = (start_id + i)%nnz;
                float r = __ldg( &R[offset].rate); //get the address of the rating field, read it from the cache
                int u = __ldg(&R[offset].u); // u & v are indices of R matrix
                int v = __ldg(&R[offset].v);
                */

                //read the p & q into register file.
                int base_p = u*k;
                int base_q = v*k;

                float tmp_p1 = __half2float(p[base_p + lane_id]);
                float tmp_q1 = __half2float(q[base_q + lane_id]);
            
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
                tmp_product += __shfl_down_sync(FULL_MASK, tmp_product, 1);

                tmp_product = __shfl_sync(FULL_MASK,tmp_product,0);

                float ruv = r - tmp_product; //get error

                 if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nerror = %f\n\n", ruv);}

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

                if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nwroteback to p %f,%f,%f,%f and wroteback to q %f,%f,%f,%f \n\n", 
                p[base_p + lane_id +  0], p[base_p + lane_id +  32], p[base_p + lane_id +  64], p[base_p + lane_id +  96],
                q[base_p + lane_id +  0], q[base_p + lane_id +  32], q[base_p + lane_id +  64], q[base_p + lane_id +  96]);}
            }    
        }
    }

    __syncthreads();
    if(threadIdx.x == DEBUG_THREAD && blockIdx.x == DEBUG_BLOCK){printf("\nThe value in index 0 is %f\n", p[0]);}
    
}

mf_node* readInputAsMF(float* input, int numRows, int numCols){
  mf_node* inputMF = (mf_node *) malloc(numRows * numCols * sizeof(mf_node));

  for(int u = 0; u < numRows; u++){
    for(int v = 0; v < numCols; v++){
      inputMF[u*numCols+v].u = u;
      inputMF[u*numCols+v].v = v;
      inputMF[u*numCols+v].r = input[u*numCols+v];
    }
  }

  return inputMF;
}



// for ease of testing, A = R, B = P, C = Q
int main(int argc, char **argv) {

//wbArg_t args;

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

	clock_t t;
  //args = wbArg_read(argc, argv);
  int numElems;


  //wbTime_start(Generic, "Importing data and creating memory on host");
  temp = (float *)wbImport(wbArg_getInputFile(args, 0), &numRRows,
                            &numRColumns);
  hostR = readInputAsMF(temp,numRRows,numRColumns);

  numElems = numRRows * numRColumns;

  for (int i = 0; i < no_r_b; i++) // initialize const array of random indices
    hostRand[i] = rand() % no_r_b;

  // hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
  //                          &numBColumns);
                            
  //@@ Set numCRows and numCColumns
  numPRows    =  numRRows;   // set to correct value
  numPColumns = K;   // set to correct value
  numQRows = K;
  numQColumns = numRColumns;
  //@@ Allocate the hostC matrix
  
  hostP = (half *)malloc((numPRows*numPColumns) * sizeof(half));
  hostQ = (half *)malloc((numQRows*numQColumns) * sizeof(half));

  // have to set to nonzero initially or the algorithm won't do anything
  for(int i = 0; i < numPRows * numPColumns; ++i)
    hostP[i] = 0.01;
  for(int i = 0; i < numQRows * numQColumns; ++i)
    hostQ[i] = 0.01;


  //wbTime_stop(Generic, "Importing data and creating memory on host");
  
  //wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceP, (numPRows*numPColumns)*sizeof(half));
  cudaMalloc((void **) &deviceQ, (numQRows*numQColumns)*sizeof(half));
  cudaMalloc((void **) &deviceR, (numRRows*numRColumns)*sizeof(mf_node));

  //wbTime_stop(GPU, "Allocating GPU memory.");

  //wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceR,hostR,(numRRows*numRColumns)*sizeof(mf_node),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceP,hostP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceQ,hostQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(RAND,hostRand, no_r_b * sizeof(int));

  //wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13.
  // Here, we perform a de facto ceil() operation on integers using integer arithmetic
  int st = 2;

  dim3 mygrid(((numRRows * numRColumns)-1)/no_r_b + 1);
  dim3 myblock(no_r_b/st);
  
  //wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  float lambda = 0.05;
  float beta = 0.3;
  float initialLearningRate = 0.08;
  float update_vector_size = 32;
  int num_iters = 10;
	//begin timing
	t = clock();
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
	double execution_time = ((double)t)/CLOCKS_PER_SEC
  //wbTime_stop(Compute, "Performing CUDA computation");

  //wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostP,deviceP,(numPRows*numPColumns)*sizeof(half),cudaMemcpyDeviceToHost);
  cudaMemcpy(hostQ,deviceQ,(numQRows*numQColumns)*sizeof(half),cudaMemcpyDeviceToHost);

  //wbTime_stop(Copy, "Copying output memory to the CPU");


//wbTime_start(GPU, "Freeing GPU Memory");
  printf("\n\n");
  for(int i = 0; i < numPRows; ++i){
    for(int j = 0; j < numPColumns; ++j){
      printf("%f ", hostP[i * numPColumns + j]);
    }
    printf("\n");
  }

  wbTime_start(GPU, "Freeing GPU Memory");

  //@@ Free the GPU memory here
  cudaFree(deviceP);
  cudaFree(deviceQ);
  cudaFree(deviceR);

  //wbTime_stop(GPU, "Freeing GPU Memory");

  //wbSolution(args, hostP, numPRows, numPColumns);

  free(hostP);
  free(hostQ);
  free(hostR);
  free(temp);
  return 0;
}
