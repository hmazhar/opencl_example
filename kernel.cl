__kernel void KERNEL_1_0(
    __global float3 *JxA, __global float3 *JyA, __global float3 *JzA, 
	__global float3 *JuA, __global float3 *JvA, __global float3 *JwA, 
    __global float3 *JxB, __global float3 *JyB, __global float3 *JzB, 
	__global float3 *JuB, __global float3 *JvB, __global float3 *JwB, 
	__global float3 *gamma,
	__global float3 *out_vel_A,
	__global float3 *out_omg_A,
	__global float3 *out_vel_B,
	__global float3 *out_omg_B,
	const unsigned int n_contact)
{
    int id = get_global_id(0);
    if (id >= n_contact){return;}

    float3 gam = gamma[id];
    float3 _JxA = JxA[id], _JyA = JyA[id], _JzA = JzA[id];
    float3 _JuA = JuA[id], _JvA = JvA[id], _JwA = JwA[id];
    float3 _JxB = JxB[id], _JyB = JyB[id], _JzB = JzB[id];
    float3 _JuB = JuB[id], _JvB = JvB[id], _JwB = JwB[id];

    out_vel_A[id] = _JxA*gam.x+_JyA*gam.y+_JzA*gam.z;
    out_omg_A[id] = _JuA*gam.x+_JvA*gam.y+_JwA*gam.z;
    out_vel_B[id] = _JxB*gam.x+_JyB*gam.y+_JzB*gam.z;
    out_omg_B[id] = _JuB*gam.x+_JvB*gam.y+_JwB*gam.z;

}