
__kernel void ShurA(
	__global float3 *JxyzA, 
	__global float3 *JuvwA, 
	__global float3 *JxyzB, 
	__global float3 *JuvwB, 
	__global float3 *gamma, 
	__global float3 *out_velocityA, 
	__global float3 *out_omegaA, 
	__global float3 *out_velocityB, 
	__global float3 *out_omegaB, 
	const unsigned int n_contact)
{
    const unsigned int id = get_global_id(0);
    if (id < n_contact){
    	float3 gam = gamma[id];
    	unsigned int c2 = n_contact*2;
    	out_velocityA[id]= JxyzA[id]*gam.x+JxyzA[id+n_contact]*gam.y+JxyzA[id+c2]*gam.z;
    	out_omegaA[id]= JuvwA[id]*gam.x+JuvwA[id+n_contact]*gam.y+JuvwA[id+c2]*gam.z;

    	out_velocityB[id]= JxyzB[id]*gam.x+JxyzB[id+n_contact]*gam.y+JxyzB[id+c2]*gam.z;
    	out_omegaB[id]= JuvwB[id]*gam.x+JuvwB[id+n_contact]*gam.y+JuvwB[id+c2]*gam.z;
    }
}


