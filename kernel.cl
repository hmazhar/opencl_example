__kernel void KERNEL_1_0(
    __global float3 *JxA, __global float3 *JyA, __global float3 *JzA, 
	__global float3 *JuA, __global float3 *JvA, __global float3 *JwA, 
    __global float3 *JxB, __global float3 *JyB, __global float3 *JzB, 
	__global float3 *JuB, __global float3 *JvB, __global float3 *JwB, 
	__global float3 *gamma,
	__global float8 *out_A,
	__global float8 *out_B,
	const unsigned int n_contact)
{
    int id = get_global_id(0);
    if (id >= n_contact){return;}

    float3 gam = gamma[id];
    
    float8 A,B,C, result;
    A.s012 = JxA[id]; //3
    A.s456 = JuA[id]; //7

    B.s012 = JyA[id]; //3
    B.s456 = JvA[id]; //7

    C.s012 = JzA[id]; //3
    C.s456 = JwA[id]; //7

    result = A*gam.x+B*gam.y+C*gam.z;
    out_A[id] = result;

    A.s012 = -JxA[id]; //3
    A.s456 = JuB[id]; //7

    B.s012 = -JyA[id]; //3
    B.s456 = JvB[id]; //7

    C.s012 = -JzA[id]; //3
    C.s456 = JwB[id]; //7
    result = A*gam.x+B*gam.y+C*gam.z;
    out_B[id] = result;


}