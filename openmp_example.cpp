
#include "omp.h"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <time.h>
using namespace std;


struct float4{
float x, y, z, w;
float4(){}
float4(float a, float b, float c, float d){
    x=a;
    y=b;
    z=c;
    w=d;
}

};

static inline float4 operator +(const float4 &rhs, const float4 &lhs) {
    return float4(rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z, rhs.w + lhs.w);
}

static inline float4 operator *(const float4 &rhs, const float &lhs) {
    return float4(rhs.x * lhs, rhs.y + lhs, rhs.z + lhs, rhs.w * lhs);
}

int main(int argc, char *argv[]){

int thread_num = 1;
if(argc>1){

    thread_num = atoi(argv[1]);
}
    unsigned int contacts = 1024000/2;
    if (argc > 2) {
        contacts = atoi(argv[2]);
    }

omp_set_num_threads(thread_num);
	// Length of vectors

    unsigned int constraints = contacts*3;
 
    // Host input vectors
    float *gamma_x, *gamma_y, *gamma_z;
// Host output vectors
    float *out_vel_x, *out_vel_y, *out_vel_z;
    float *out_omg_x, *out_omg_y, *out_omg_z;
 

    // Size, in bytes, of each vector
    size_t bytes = contacts*sizeof(float);
 
    // Allocate memory for each vector on host
    float4 * JxA = (float4*) malloc(contacts * sizeof(float4));
    float4 * JyA = (float4*) malloc(contacts * sizeof(float4));
    float4 * JzA = (float4*) malloc(contacts * sizeof(float4));

    float4 * JuA = (float4*) malloc(contacts * sizeof(float4));
    float4 * JvA = (float4*) malloc(contacts * sizeof(float4));
    float4 * JwA = (float4*) malloc(contacts * sizeof(float4));

    float4 * JxB = (float4*) malloc(contacts * sizeof(float4));
    float4 * JyB = (float4*) malloc(contacts * sizeof(float4));
    float4 * JzB = (float4*) malloc(contacts * sizeof(float4));

    float4 * JuB = (float4*) malloc(contacts * sizeof(float4));
    float4 * JvB = (float4*) malloc(contacts * sizeof(float4));
    float4 * JwB = (float4*) malloc(contacts * sizeof(float4));

    float4 * h_g = (float4*) malloc(contacts * sizeof(float4));

    float4 * out_vel_A = (float4*) malloc(contacts * sizeof(float4));
    float4 * out_omg_A = (float4*) malloc(contacts * sizeof(float4));
    float4 * out_vel_B = (float4*) malloc(contacts * sizeof(float4));
    float4 * out_omg_B = (float4*) malloc(contacts * sizeof(float4));
 
    // Initialize vectors on host
    int i;
    for (i = 0; i < contacts; i++) {
        JxA[i].x = sinf(i) * sinf(i);
        JxA[i].y = sinf(i) * sinf(i);
        JxA[i].z = sinf(i) * sinf(i);

        JyA[i].x = sinf(i) * sinf(i);
        JyA[i].y = sinf(i) * sinf(i);
        JyA[i].z = sinf(i) * sinf(i);

        JzA[i].x = sinf(i) * sinf(i);
        JzA[i].y = sinf(i) * sinf(i);
        JzA[i].z = sinf(i) * sinf(i);

        h_g[i].x = sinf(i) * sinf(i);
        h_g[i].y = sinf(i) * sinf(i);
        h_g[i].z = sinf(i) * sinf(i);

        JuA[i].x = sinf(i) * sinf(i);
        JuA[i].y = sinf(i) * sinf(i);
        JuA[i].z = sinf(i) * sinf(i);

        JvA[i].x = cosf(i) * cosf(i);
        JvA[i].y = cosf(i) * cosf(i);
        JvA[i].z = cosf(i) * cosf(i);

        JwA[i].x = cosf(i) * cosf(i);
        JwA[i].y = cosf(i) * cosf(i);
        JwA[i].z = cosf(i) * cosf(i);

        JxB[i].x = sinf(i) * sinf(i);
        JxB[i].y = sinf(i) * sinf(i);
        JxB[i].z = sinf(i) * sinf(i);


        JyB[i].x = cosf(i) * cosf(i);
        JyB[i].y = cosf(i) * cosf(i);
        JyB[i].z = cosf(i) * cosf(i);


        JzB[i].x = cosf(i) * cosf(i);
        JzB[i].y = cosf(i) * cosf(i);
        JzB[i].z = cosf(i) * cosf(i);



        JuB[i].x = sinf(i) * sinf(i);
        JuB[i].y = sinf(i) * sinf(i);
        JuB[i].z = sinf(i) * sinf(i);


        JvB[i].x = cosf(i) * cosf(i);
        JvB[i].y = cosf(i) * cosf(i);
        JvB[i].z = cosf(i) * cosf(i);


        JwB[i].x = cosf(i) * cosf(i);
        JwB[i].y = cosf(i) * cosf(i);
        JwB[i].z = cosf(i) * cosf(i);


    }
 
int n_contact = contacts;


double total_time_omp;
double total_flops;
double total_memory;
double  runs = 100;
for(int i=0; i<runs; i++){

double start = omp_get_wtime();
#pragma omp parallel for
for(int id=0; id<n_contact; id++){

    float4 gam = h_g[id];
    float4 _JxA = JxA[id], _JyA = JyA[id], _JzA = JzA[id];
    float4 _JuA = JuA[id], _JvA = JvA[id], _JwA = JwA[id];
    float4 _JxB = JxB[id], _JyB = JyB[id], _JzB = JzB[id];
    float4 _JuB = JuB[id], _JvB = JvB[id], _JwB = JwB[id];

    out_vel_A[id] = _JxA*gam.x+_JyA*gam.y+_JzA*gam.z;
    out_omg_A[id] = _JuA*gam.x+_JvA*gam.y+_JwA*gam.z;
    out_vel_B[id] = _JxB*gam.x+_JyB*gam.y+_JzB*gam.z;
    out_omg_B[id] = _JuB*gam.x+_JvB*gam.y+_JwB*gam.z;

}
double end = omp_get_wtime();

    total_time_omp += (end - start) * 1000;
    total_flops += 60*contacts/((end - start))/1e9;
    total_memory+= 204*contacts/((end - start) )/1024.0/1024.0/1024.0;

}

    printf("\nExecution time in milliseconds =  %0.3f ms | %0.3f Gflop | %0.3f GB/s \n", total_time_omp/runs, total_flops/runs, total_memory/runs);

    //release host memory
    free(JxA);
    free(JyA);
    free(JzA);

    free(JuA);
    free(JvA);
    free(JwA);

    free(JxB);
    free(JyB);
    free(JzB);

    free(JuB);
    free(JvB);
    free(JwB);

    free(h_g);

    free(out_vel_A);
    free(out_omg_A);
    free(out_vel_B);
    free(out_omg_B);
 
    return 0;
}
