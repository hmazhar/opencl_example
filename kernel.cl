__kernel void KERNEL_1_0(
    __global float *JxA, __global float *JyA, __global float *JzA, 
	__global float *JuA, __global float *JvA, __global float *JwA, 
    __global float *JxB, __global float *JyB, __global float *JzB, 
	__global float *JuB, __global float *JvB, __global float *JwB, 
	__global float *gamma_x, __global float *gamma_y, __global float *gamma_z,
	__global float *out_vel_xA, __global float *out_vel_yA, __global float *out_vel_zA,
	__global float *out_omg_xA, __global float *out_omg_yA, __global float *out_omg_zA,
	__global float *out_vel_xB, __global float *out_vel_yB, __global float *out_vel_zB,
	__global float *out_omg_xB, __global float *out_omg_yB, __global float *out_omg_zB,
	const unsigned int n_contact)
{
    int id = get_global_id(0);
    if (id >= n_contact){return;}

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