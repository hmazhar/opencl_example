
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
    size_t bytes = constraints*sizeof(float);
 
    // Allocate memory for each vector on host
    float * JxA              = (float*) malloc(bytes);
    float * JyA              = (float*) malloc(bytes);
    float * JzA              = (float*) malloc(bytes);

    float * JuA              = (float*) malloc(bytes);
    float * JvA              = (float*) malloc(bytes);
    float * JwA              = (float*) malloc(bytes);

    float * JxB              = (float*) malloc(bytes);
    float * JyB              = (float*) malloc(bytes);
    float * JzB              = (float*) malloc(bytes);

    float * JuB              = (float*) malloc(bytes);
    float * JvB              = (float*) malloc(bytes);
    float * JwB              = (float*) malloc(bytes);

    gamma_x         = (float*) malloc(bytes);
    gamma_y         = (float*) malloc(bytes);
    gamma_z         = (float*) malloc(bytes);

    float * out_vel_xA  = (float*) malloc(bytes);
    float * out_vel_yA  = (float*) malloc(bytes);
    float * out_vel_zA  = (float*) malloc(bytes);

    float * out_omg_xA     = (float*) malloc(bytes);
    float * out_omg_yA     = (float*) malloc(bytes);
    float * out_omg_zA     = (float*) malloc(bytes);


    float * out_vel_xB  = (float*) malloc(bytes);
    float * out_vel_yB  = (float*) malloc(bytes);
    float * out_vel_zB  = (float*) malloc(bytes);

    float * out_omg_xB     = (float*) malloc(bytes);
    float * out_omg_yB     = (float*) malloc(bytes);
    float * out_omg_zB     = (float*) malloc(bytes);
 
    // Initialize vectors on host
    int i;
for (i = 0; i < constraints; i++) {
        JxA[i] = sinf(i) * sinf(i);
        JyA[i] = cosf(i) * cosf(i);
        JzA[i] = cosf(i) * cosf(i);

        JuA[i] = sinf(i) * sinf(i);
        JvA[i] = cosf(i) * cosf(i);
        JwA[i] = cosf(i) * cosf(i);

        JxB[i] = sinf(i) * sinf(i);
        JyB[i] = cosf(i) * cosf(i);
        JzB[i] = cosf(i) * cosf(i);

        JuB[i] = sinf(i) * sinf(i);
        JvB[i] = cosf(i) * cosf(i);
        JwB[i] = cosf(i) * cosf(i);

        gamma_x[i] = sinf(i) * sinf(i);
        gamma_y[i] = cosf(i) * cosf(i);
        gamma_z[i] = cosf(i) * cosf(i);

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

    float gam_x = gamma_x[id];
    float gam_y = gamma_y[id];
    float gam_z = gamma_z[id];

    out_vel_xA[id] = JxA[id+n_contact*0]*gam_x+JxA[id+n_contact*1]*gam_y+JxA[id+n_contact*2]*gam_z;
    out_vel_yA[id] = JyA[id+n_contact*0]*gam_x+JyA[id+n_contact*1]*gam_y+JyA[id+n_contact*2]*gam_z;
    out_vel_zA[id] = JzA[id+n_contact*0]*gam_x+JzA[id+n_contact*1]*gam_y+JzA[id+n_contact*2]*gam_z;
 
    out_omg_xA[id] = JuA[id+n_contact*0]*gam_x+JuA[id+n_contact*1]*gam_y+JuA[id+n_contact*2]*gam_z;
    out_omg_yA[id] = JvA[id+n_contact*0]*gam_x+JvA[id+n_contact*1]*gam_y+JvA[id+n_contact*2]*gam_z;
    out_omg_zA[id] = JwA[id+n_contact*0]*gam_x+JwA[id+n_contact*1]*gam_y+JwA[id+n_contact*2]*gam_z;
 
    out_vel_xB[id] = JxB[id+n_contact*0]*gam_x+JxB[id+n_contact*1]*gam_y+JxB[id+n_contact*2]*gam_z;
    out_vel_yB[id] = JyB[id+n_contact*0]*gam_x+JyB[id+n_contact*1]*gam_y+JyB[id+n_contact*2]*gam_z;
    out_vel_zB[id] = JzB[id+n_contact*0]*gam_x+JzB[id+n_contact*1]*gam_y+JzB[id+n_contact*2]*gam_z;
 
    out_omg_xB[id] = JuB[id+n_contact*0]*gam_x+JuB[id+n_contact*1]*gam_y+JuB[id+n_contact*2]*gam_z;
    out_omg_yB[id] = JvB[id+n_contact*0]*gam_x+JvB[id+n_contact*1]*gam_y+JvB[id+n_contact*2]*gam_z;
    out_omg_zB[id] = JwB[id+n_contact*0]*gam_x+JwB[id+n_contact*1]*gam_y+JwB[id+n_contact*2]*gam_z;

}
double end = omp_get_wtime();

    total_time_omp += (end - start) * 1000;
    total_flops += 60*contacts/((end - start))/1e9;
    total_memory+= 204*contacts/((end - start) )/1024.0/1024.0/1024.0;

}

    printf("\nExecution time in milliseconds =  %0.3f ms | %0.3f Gflop | %0.3f GB/s \n", total_time_omp/runs, total_flops/runs, total_memory/runs);

    //release host memory
    free(JxA           );
    free(JyA           );
    free(JzA           );

    free(JuA           );
    free(JvA           );
    free(JwA           );

    free(JxB           );
    free(JyB           );
    free(JzB           );

    free(JuB           );
    free(JvB           );
    free(JwB           );

    free(gamma_x       );
    free(gamma_y       );
    free(gamma_z       );

    free(out_vel_xA);
    free(out_vel_yA);
    free(out_vel_zA);

    free(out_omg_xA   );
    free(out_omg_yA   );
    free(out_omg_zA   );

    free(out_vel_xB);
    free(out_vel_yB);
    free(out_vel_zB);

    free(out_omg_xB   );
    free(out_omg_yB   );
    free(out_omg_zB   );
 
    return 0;
}
