#ifndef utils
#define utils
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>
struct mf_node{
	int u,v;
	short rate;
};

/*
float toFloat(uint8_t x) {
    return x / 255.0e7;
}

uint8_t toHalf(float x) {
    if (x < 0) return 0;
    if (x > 1e-7) return 255;
    return 255.0e7 * x; // this truncates; add 0.5 to round instead
}
*/
mf_node* readInputAsMF(half* input, int numRows, int numCols){
  mf_node* inputMF = (mf_node *) malloc(numRows * numCols * sizeof(mf_node));

  for(int u = 0; u < numRows; u++){
    for(int v = 0; v < numCols; v++){
      inputMF[u*numCols+v].u = u;
      inputMF[u*numCols+v].v = v;
      inputMF[u*numCols+v].rate = input[u*numCols+v];
    }
  }
  return inputMF;
}

void printMF_node(mf_node* node, int numRows, int numCols){
  for(int u = 0; u < numRows; u++){
    for(int v = 0; v < numCols; v++){
        int u_ = (node[u*numCols+v].u);
        int v_ = (node[u*numCols+v].v);
        float rate = (node[u*numCols+v].rate);
        printf( "{(%d, %d) %.2f}  " , u_, v_, rate);
    }
      printf("\n");
  }
    printf("\n");
}


static float* generate_data(int height, int width) {
    float* data = (float *)malloc(sizeof(float) * width * height);
  int i;
  for (i = 0; i < width * height; i++) {
    data[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }
  return data;
}

void print(float* input, int numRows, int numCols){
    for (int i = 0; i < numRows; i++ ){
        for (int j = 0; j < numCols; j++ ){
            printf("%.2f ", input[i*numCols+ j]);

        }
    }
}

#endif
