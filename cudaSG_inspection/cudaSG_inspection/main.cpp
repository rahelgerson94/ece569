#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "utils.h"
using std:: cout;
using std::endl;
struct bla{
    int a[6];
    int b[6];
    float c[6];
};
int main(int argc, char**argv){
//    uint8_t  *hostP; // The A matrix
//    uint8_t  *hostQ; // The B matrix
//
//    mf_node *hostR; // The output C matrix
//    int numPRows;    // number of rows in the matrix A
//    int numPColumns; // number of columns in the matrix A
//    int numQRows;    // number of rows in the matrix B
//    int numQColumns; // number of columns in the matrix B
//    int numRRows;    // number of rows in the matrix C(you have to set this)
//    int numRColumns; // number of columns in the matrix C (you have to set
//                   // this)
//    numRRows = 10;
//    numRColumns = 4;
//    int K = 5;
//
//    numPRows    =  numRRows;
//    numPColumns = K;
//    numQRows = K;
//    numQColumns = numRColumns;
//    //allocate and populate
//    hostP = generate_data(numPRows, numPColumns);
//    hostQ = generate_data(numQRows, numQColumns);
//    //populate R
//    uint8_t * tmp = (uint8_t *)malloc(numRRows*numRColumns*sizeof(uint8_t ));
//
//    hostR = readInputAsMF(tmp,numRRows,numRColumns); //populate
//    printMF_node(hostR, numRRows, numRColumns);
//
    bla bla;
    for (int i = 0; i < 6 ;i++){
        bla.a[i] = i;
    }
    for (int i = 0; i < 6 ;i++){
        bla.b[i] = i+6;
    }
    for (int i = 0; i < 6 ;i++){
        bla.c[i] = i+12;
    }
    int index = 1;
    cout<< bla.a[index] <<endl;
    cout<< bla.b[index] <<endl;
     
    
    return 0;
    
}

