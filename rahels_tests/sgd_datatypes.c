#include "sgd_datatypes.h"
struct mf_node{
	int u,v;
	float rate;
};

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
