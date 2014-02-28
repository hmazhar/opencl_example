__kernel void KERNEL_1_0(
    __global float3 *JxA, __global float3 *JyA, __global float3 *JzA, 
	__global float3 *JuA, __global float3 *JvA, __global float3 *JwA, 
    __global float3 *JxB, __global float3 *JyB, __global float3 *JzB, 
	__global float3 *JuB, __global float3 *JvB, __global float3 *JwB, 
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
    float3 _JxA = JxA[id], _JyA = JyA[id], _JzA = JzA[id];
    float3 _JuA = JuA[id], _JvA = JvA[id], _JwA = JwA[id];
    float3 _JxB = JxB[id], _JyB = JyB[id], _JzB = JzB[id];
    float3 _JuB = JuB[id], _JvB = JvB[id], _JwB = JwB[id];

    out_vel_xA[id] = _JxA.x*gam.x+_JyA.x*gam.y+_JzA.x*gam.z;
    out_vel_yA[id] = _JxA.y*gam.x+_JyA.y*gam.y+_JzA.y*gam.z;
    out_vel_zA[id] = _JxA.z*gam.x+_JyA.z*gam.y+_JzA.z*gam.z;
 
    out_omg_xA[id] = _JuA.x*gam.x+_JvA.x*gam.y+_JwA.x*gam.z;
    out_omg_yA[id] = _JuA.y*gam.x+_JvA.y*gam.y+_JwA.y*gam.z;
    out_omg_zA[id] = _JuA.z*gam.x+_JvA.z*gam.y+_JwA.z*gam.z;
 
    out_vel_xB[id] = _JxB.x*gam.x+_JyB.x*gam.y+_JzB.x*gam.z;
    out_vel_yB[id] = _JxB.y*gam.x+_JyB.y*gam.y+_JzB.y*gam.z;
    out_vel_zB[id] = _JxB.z*gam.x+_JyB.z*gam.y+_JzB.z*gam.z;
 
    out_omg_xB[id] = _JuB.x*gam.x+_JvB.x*gam.y+_JwB.x*gam.z;
    out_omg_yB[id] = _JuB.y*gam.x+_JvB.y*gam.y+_JwB.y*gam.z;
    out_omg_zB[id] = _JuB.z*gam.x+_JvB.z*gam.y+_JwB.z*gam.z;
}