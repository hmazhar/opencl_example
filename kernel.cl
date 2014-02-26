
__kernel void ShurA(
	__global float16 *JxyzA, 
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


    	float16 C1 = JxyzA[id];
    	float16 C2 = JxyzA[id+n_contact];
    	float16 C3 = JxyzA[id+n_contact*2];
    	float3 outvA, outvB, outoA, outoB;

    	outvA.x =  C1.s0*gam.x+C2.s0*gam.y+C3.s0*gam.z;
    	outvA.y =  C1.s1*gam.x+C2.s1*gam.y+C3.s1*gam.z;
    	outvA.z =  C1.s2*gam.x+C2.s2*gam.y+C3.s2*gam.z;
    	//outvA.w =  C1.s3*gam.x+C2.s3*gam.y+C3.s3*gam.z;

    	outoA.x =  C1.s4*gam.x+C2.s4*gam.y+C3.s4*gam.z;
    	outoA.y =  C1.s5*gam.x+C2.s5*gam.y+C3.s5*gam.z;
    	outoA.z =  C1.s6*gam.x+C2.s6*gam.y+C3.s6*gam.z;
    	//outoA.w =  C1.s7*gam.x+C2.s7*gam.y+C3.s7*gam.z;


    	outvB.x =  C1.s8 *gam.x+C2.s8 *gam.y+C3.s8 *gam.z;
    	outvB.y =  C1.s9 *gam.x+C2.s9 *gam.y+C3.s9 *gam.z;
    	outvB.z =  C1.sa*gam.x+C2.sa*gam.y+C3.sa*gam.z;
    	//outvB.w =  C1.sb*gam.x+C2.sb*gam.y+C3.sb*gam.z;

    	outoB.x =  C1.sc*gam.x+C2.sc*gam.y+C3.sc*gam.z;
    	outoB.y =  C1.sd*gam.x+C2.sd*gam.y+C3.sd*gam.z;
    	outoB.z =  C1.se*gam.x+C2.se*gam.y+C3.se*gam.z;
    	//outoB.w =  C1.sf*gam.x+C2.sf*gam.y+C3.sf*gam.z;


    	out_velocityA[id]= outvA;
    	out_omegaA[id]= outoA;

    	out_velocityB[id]= outvB;
    	out_omegaB[id]= outoB;
    }
}


