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

    float16 A;
    A.s012 = _JxA; //3
    A.s456 = _JuA; //7
    A.s89a = _JxB; //b
    A.scde = _JuB; //f

    float16 B;
    B.s012 = _JyA; //3
    B.s456 = _JvA; //7
    B.s89a = _JyB; //b
    B.scde = _JvB; //f


    float16 C;
    C.s012 = _JzA; //3
    C.s456 = _JwA; //7
    C.s89a = _JzB; //b
    C.scde = _JwB; //f

    float16 result = A*gam.x+B*gam.y+C*gam.z;

    out_vel_A[id] = result.s012;
    out_omg_A[id] = result.s456;
    out_vel_B[id] = result.s89a;
    out_omg_B[id] = result.scde;

}