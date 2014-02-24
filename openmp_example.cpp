
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
    unsigned int n = 10240000;
 
    // Host input vectors
    double *a;
    double *b;
    // Host output vector
    double *c;
 

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c = (double*)malloc(bytes);
 
    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        a[i] = sinf(i)*sinf(i);
        b[i] = cosf(i)*cosf(i);
    }
 

double start = omp_get_wtime();
#pragma omp parallel for
for(int id=0; id<n; id++){
        for(int i=0; i<100; i++){
double _a = a[id] + b[id] * b[id] +(a[id] * b[id])/b[id];
double _b = a[id] - b[id] / b[id] -(a[id] / b[id])*b[id];
double _c = b[id] + a[id] * a[id] +(b[id] * a[id])/b[id];
double _d = b[id] - a[id] / b[id] -(b[id] / b[id])*a[id];
c[id] = _a + _b * _c +(_a * _d)/_c;
}
}
double end = omp_get_wtime();
printf("Time: \t %f \n", (end-start)*1000); 

    //release host memory
    free(a);
    free(b);
    free(c);
 
    return 0;
}
