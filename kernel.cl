
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
    	out_velocityA[id]= JxyzA[id]*gam.x+JxyzA[id+n_contact*1]*gam.y+JxyzA[id+n_contact*2]*gam.z;
    	out_omegaA[id]= JxyzA[id+n_contact*3]*gam.x+JxyzA[id+n_contact*4]*gam.y+JxyzA[id+n_contact*5]*gam.z;

    	out_velocityB[id]= JxyzA[id+n_contact*6]*gam.x+JxyzA[id+n_contact*7]*gam.y+JxyzA[id+n_contact*8]*gam.z;
    	out_omegaB[id]= JxyzA[id+n_contact*9]*gam.x+JxyzA[id+n_contact*10]*gam.y+JxyzA[id+n_contact*11]*gam.z;
    }
}


3+9*4



