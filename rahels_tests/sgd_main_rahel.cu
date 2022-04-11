#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>

#include <stdexcept>
#include <string>
#include <vector>
#include "utils.cu" //includes fp16, iostream

#include <time.h>

//#define K 128
//#define no_r_b 1024 //
#define FULL_MASK 0xffffffff
#define DEBUG_THREAD 511
#define DEBUG_BLOCK 5
__constant__ int RAND[no_r_b];

int main(int argc, char**argv){
    
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
    numRRows = 10;
    numRColumns = 4;
    int K = 5;
   
    numPRows    =  numRRows;
    numPColumns = K;
    numQRows = K;
    numQColumns = numRColumns;
    //allocate and populate
    hostP = generate_data(numPRows, numPColumns);
    hostQ = generate_data(numQRows, numQColumns);
    //populate R
    half * tmp = (half *)malloc(numRRows*numRColumns*sizeof(half ));
    
    hostR = readInputAsMF(tmp,numRRows,numRColumns); //populate
    printMF_node(hostR, numRRows, numRColumns);
    
    /*
     float x = 255.6;
    half shortX = toHalf(x);
    printf("%.2f\n", x);
    printf("%.2f\n", shortX);
     */
	
    
    return 0;
    
}

