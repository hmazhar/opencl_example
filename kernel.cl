
__kernel void KERNEL_1_0(
    __global float3 *norm, 
	__global float3 *JuA, __global float3 *JvA, __global float3 *JwA, 
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


    float3 U = norm[id], V, W;
	W = cross(U, (float3)(0, 1, 0));
	float mzlen = length(W);

	if (mzlen < 0.0001f) { 
		float3 mVsingular = (float3)(1, 0, 0);
		W = cross(U, mVsingular);
		mzlen = length(W);
	}
	W = W * 1.0f / mzlen;
	V = cross(W, U);

    float3 gam = gamma[id];
    
    float8 A,B,C, result;
    A.s012 = -U; //3
    A.s456 = JuA[id]; //7

    B.s012 = -V; //3
    B.s456 = JvA[id]; //7

    C.s012 = -W; //3
    C.s456 = JwA[id]; //7

    result = A*gam.x+B*gam.y+C*gam.z;
    out_vel_A[id] = result.s012;
    out_omg_A[id] = result.s456;

    A.s012 = U; //3
    A.s456 = JuB[id]; //7

    B.s012 = V; //3
    B.s456 = JvB[id]; //7

    C.s012 = W; //3
    C.s456 = JwB[id]; //7
    result = A*gam.x+B*gam.y+C*gam.z;
    out_vel_B[id] = result.s012;
    out_omg_B[id] = result.s456;


}