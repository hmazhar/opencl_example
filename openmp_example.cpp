
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
    float *out_velocity_x, *out_velocity_y, *out_velocity_z;
    float *out_omega_x, *out_omega_y, *out_omega_z;
 

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

    float * out_velocity_xA  = (float*) malloc(bytes);
    float * out_velocity_yA  = (float*) malloc(bytes);
    float * out_velocity_zA  = (float*) malloc(bytes);

    float * out_omega_xA     = (float*) malloc(bytes);
    float * out_omega_yA     = (float*) malloc(bytes);
    float * out_omega_zA     = (float*) malloc(bytes);


    float * out_velocity_xB  = (float*) malloc(bytes);
    float * out_velocity_yB  = (float*) malloc(bytes);
    float * out_velocity_zB  = (float*) malloc(bytes);

    float * out_omega_xB     = (float*) malloc(bytes);
    float * out_omega_yB     = (float*) malloc(bytes);
    float * out_omega_zB     = (float*) malloc(bytes);
 
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
double start = omp_get_wtime();
#pragma omp parallel for
for(int id=0; id<n_contact; id++){

        float gam_x = gamma_x[id];
        float gam_y = gamma_y[id];
        float gam_z = gamma_z[id];
        unsigned int c2 = n_contact*2;
        out_velocity_xA[id]= JxA[id]*gam_x+JxA[id+n_contact]*gam_y+JxA[id+c2]*gam_z;
        out_velocity_yA[id]= JyA[id]*gam_x+JyA[id+n_contact]*gam_y+JyA[id+c2]*gam_z;
        out_velocity_zA[id]= JzA[id]*gam_x+JzA[id+n_contact]*gam_y+JzA[id+c2]*gam_z;

        out_omega_xA[id]= JuA[id]*gam_x+JuA[id+n_contact]*gam_y+JuA[id+c2]*gam_z;
        out_omega_yA[id]= JvA[id]*gam_x+JvA[id+n_contact]*gam_y+JvA[id+c2]*gam_z;
        out_omega_zA[id]= JwA[id]*gam_x+JwA[id+n_contact]*gam_y+JwA[id+c2]*gam_z;

        out_velocity_xB[id]= JxB[id]*gam_x+JxB[id+n_contact]*gam_y+JxB[id+c2]*gam_z;
        out_velocity_yB[id]= JyB[id]*gam_x+JyB[id+n_contact]*gam_y+JyB[id+c2]*gam_z;
        out_velocity_zB[id]= JzB[id]*gam_x+JzB[id+n_contact]*gam_y+JzB[id+c2]*gam_z;

        out_omega_xB[id]= JuB[id]*gam_x+JuB[id+n_contact]*gam_y+JuB[id+c2]*gam_z;
        out_omega_yB[id]= JvB[id]*gam_x+JvB[id+n_contact]*gam_y+JvB[id+c2]*gam_z;
        out_omega_zB[id]= JwB[id]*gam_x+JwB[id+n_contact]*gam_y+JwB[id+c2]*gam_z;

}
double end = omp_get_wtime();
printf("Time: \t %f \t %f Gflops\n", (end-start)*1000, (85*n_contact)/((end-start))/(1e9)); 

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

    free(out_velocity_xA);
    free(out_velocity_yA);
    free(out_velocity_zA);

    free(out_omega_xA   );
    free(out_omega_yA   );
    free(out_omega_zA   );

    free(out_velocity_xB);
    free(out_velocity_yB);
    free(out_velocity_zB);

    free(out_omega_xB   );
    free(out_omega_yB   );
    free(out_omega_zB   );
 
    return 0;
}
