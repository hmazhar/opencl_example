
#include "omp.h"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <time.h>
using namespace std;


int main(int argc, char *argv[]){

int thread_num = 1;
if(argc>1){

    thread_num = atoi(argv[1]);
}

omp_set_num_threads(thread_num);
	// Length of vectors
    unsigned int n = 102400000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;
 

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = sinf(i)*sinf(i);
        h_b[i] = cosf(i)*cosf(i);
    }
 

double start = omp_get_wtime();
#pragma omp parallel for
for(int i=0; i<n; i++){
	h_c[i] = h_a[i] + h_b[i] * h_b[i] +(h_a[i] * h_b[i])/h_b[i];
}
double end = omp_get_wtime();
printf("Time: \t %f \n", (end-start)*1000); 

    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
