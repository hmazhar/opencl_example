
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
    unsigned int contacts = 1024000*3;
    unsigned int constraints = contacts*3;
 
    // Host input vectors
    float *Jx, *Jy, *Jz;
    float *Ju, *Jv, *Jw;
    float *gamma_x, *gamma_y, *gamma_z;
// Host output vectors
    float *out_velocity_x, *out_velocity_y, *out_velocity_z;
    float *out_omega_x, *out_omega_y, *out_omega_z;
 

    // Size, in bytes, of each vector
    size_t bytes = constraints*sizeof(float);
 
    // Allocate memory for each vector on host
    Jx              = (float*) malloc(bytes);
    Jy              = (float*) malloc(bytes);
    Jz              = (float*) malloc(bytes);

    Ju              = (float*) malloc(bytes);
    Jv              = (float*) malloc(bytes);
    Jw              = (float*) malloc(bytes);

    gamma_x         = (float*) malloc(bytes);
    gamma_y         = (float*) malloc(bytes);
    gamma_z         = (float*) malloc(bytes);

    out_velocity_x  = (float*) malloc(bytes);
    out_velocity_y  = (float*) malloc(bytes);
    out_velocity_z  = (float*) malloc(bytes);

    out_omega_x     = (float*) malloc(bytes);
    out_omega_y     = (float*) malloc(bytes);
    out_omega_z     = (float*) malloc(bytes);
 
    // Initialize vectors on host
    int i;
for (i = 0; i < constraints; i++) {
        Jx[i] = sinf(i) * sinf(i);
        Jy[i] = cosf(i) * cosf(i);
        Jz[i] = cosf(i) * cosf(i);

        Ju[i] = sinf(i) * sinf(i);
        Jv[i] = cosf(i) * cosf(i);
        Jw[i] = cosf(i) * cosf(i);

        gamma_x[i] = sinf(i) * sinf(i);
        gamma_y[i] = cosf(i) * cosf(i);
        gamma_z[i] = cosf(i) * cosf(i);

    }
 
int n_contact = contacts;
double start = omp_get_wtime();
#pragma omp parallel for
for(int id=0; id<n_contact; id++){

        float gam_x = gamma_x[id];
        float gam_y = gamma_y[id];
        float gam_z = gamma_z[id];

        out_velocity_x[id]= Jx[id+n_contact*0]*gam_x+Jx[id+n_contact*1]*gam_y+Jx[id+n_contact*2]*gam_z;
        out_velocity_y[id]= Jy[id+n_contact*0]*gam_x+Jy[id+n_contact*1]*gam_y+Jy[id+n_contact*2]*gam_z;
        out_velocity_z[id]= Jz[id+n_contact*0]*gam_x+Jz[id+n_contact*1]*gam_y+Jz[id+n_contact*2]*gam_z;

        out_omega_x[id]= Ju[id+n_contact*0]*gam_x+Ju[id+n_contact*1]*gam_y+Ju[id+n_contact*2]*gam_z;
        out_omega_y[id]= Jv[id+n_contact*0]*gam_x+Jv[id+n_contact*1]*gam_y+Jv[id+n_contact*2]*gam_z;
        out_omega_z[id]= Jw[id+n_contact*0]*gam_x+Jw[id+n_contact*1]*gam_y+Jw[id+n_contact*2]*gam_z;

}
double end = omp_get_wtime();
printf("Time: \t %f \n", (end-start)*1000); 

    //release host memory
    free(Jx            );
    free(Jy            );
    free(Jz            );

    free(Ju            );
    free(Jv            );
    free(Jw            );

    free(gamma_x       );
    free(gamma_y       );
    free(gamma_z       );

    free(out_velocity_x);
    free(out_velocity_y);
    free(out_velocity_z);

    free(out_omega_x   );
    free(out_omega_y   );
    free(out_omega_z   );
 
    return 0;
}
