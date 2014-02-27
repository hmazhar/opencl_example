
__kernel void ShurA(
	__global float16 *JxyzA,
	__global float3 *gamma, 
	__global float16 *out_velocityA, 
	const unsigned int n_contact)
{
    const unsigned int id = get_global_id(0);
    if (id < n_contact){
    	float3 gam = gamma[id];
    	
    	float16 C1 = JxyzA[id];
    	float16 C2 = JxyzA[id+n_contact];
    	float16 C3 = JxyzA[id+n_contact*2];

    	float16 result = C1*gam.x+C2*gam.y+C3*gam.z;
    	out_velocityA[id]= result;
    }
}


