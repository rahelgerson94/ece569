#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

#include <stdio.h>
#include <cuda_fp16.h>
#include "utils.h"
#define DSIZE 4

#define nTPB 256
__global__ void convert2half(float *din, float *dout, int numRows, int numCols){
	int col = threadIdx.x+blockDim.x*blockIdx.x;
	int row = threadIdx.y+blockDim.y*blockIdx.y;
  if (row < numRows && col < numCols){
    half kin = __float2half(din[row*numCols + col]);
    half kout;
#if __CUDA_ARCH__ >= 530
    kout = __hmul(kin, scf);
#else
    kout = __float2half(__half2float(kin)*__half2float(scf));
#endif
    dout[idx] = __half2float(kout);
    }
}

int main(){

  float *hin, *hout, *din, *dout;
  hin  = (float *)malloc(DSIZE*sizeof(float));
  hout = (float *)malloc(DSIZE*sizeof(float));
  for (int i = 0; i < DSIZE; i++) 
		hin[i] = i;

  cudaMalloc(&din,  DSIZE*sizeof(float));
  cudaMalloc(&dout, DSIZE*sizeof(float));
  cudaMemcpy(din, hin, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  half_scale_convert2half<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(din, dout, DSIZE);
  cudaMemcpy(hout, dout, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  print(
  return 0;
}

