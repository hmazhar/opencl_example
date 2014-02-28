__kernel void KERNEL_1_0(
    __global float3 *JxA, __global float3 *JyA, __global float3 *JzA, 
	__global float3 *JuA, __global float3 *JvA, float3 float *JwA, 
    __global float *JxB, __global float *JyB, __global float *JzB, 
	__global float *JuB, __global float *JvB, __global float *JwB, 
	__global float3 *gamma,
	__global float *out_vel_xA, __global float *out_vel_yA, __global float *out_vel_zA,
	__global float *out_omg_xA, __global float *out_omg_yA, __global float *out_omg_zA,
	__global float *out_vel_xB, __global float *out_vel_yB, __global float *out_vel_zB,
	__global float *out_omg_xB, __global float *out_omg_yB, __global float *out_omg_zB,
	const unsigned int n_contact)
{
    int id = get_global_id(0);
    if (id >= n_contact){return;}

    float3 gam = gamma[id];

    out_vel_xA[id] = JxA[id].x*gam.x+JyA[id].x*gam.y+JzA[id].x*gam.z;
    out_vel_yA[id] = JxA[id].y*gam.x+JyA[id].y*gam.y+JzA[id].y*gam.z;
    out_vel_zA[id] = JxA[id].z*gam.x+JyA[id].z*gam.y+JzA[id].z*gam.z;
 
    out_omg_xA[id] = JuA[id].x*gam.x+JvA[id].x*gam.y+JwA[id].x*gam.z;
    out_omg_yA[id] = JuA[id].y*gam.x+JvA[id].y*gam.y+JwA[id].y*gam.z;
    out_omg_zA[id] = JuA[id].z*gam.x+JvA[id].z*gam.y+JwA[id].z*gam.z;
 
    out_vel_xB[id] = JxB[id+n_contact*0]*gam.x+JyB[id+n_contact*0]*gam.y+JzB[id+n_contact*0]*gam.z;
    out_vel_yB[id] = JxB[id+n_contact*1]*gam.x+JyB[id+n_contact*1]*gam.y+JzB[id+n_contact*1]*gam.z;
    out_vel_zB[id] = JxB[id+n_contact*2]*gam.x+JyB[id+n_contact*2]*gam.y+JzB[id+n_contact*2]*gam.z;
 
    out_omg_xB[id] = JuB[id+n_contact*0]*gam.x+JvB[id+n_contact*0]*gam.y+JwB[id+n_contact*0]*gam.z;
    out_omg_yB[id] = JuB[id+n_contact*1]*gam.x+JvB[id+n_contact*1]*gam.y+JwB[id+n_contact*1]*gam.z;
    out_omg_zB[id] = JuB[id+n_contact*2]*gam.x+JvB[id+n_contact*2]*gam.y+JwB[id+n_contact*2]*gam.z;
}