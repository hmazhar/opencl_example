
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
    	float4 outvA, outvB, outoA, outoB;

    	outvA.x =  C1.s0*gam.x+C2.s0*gam.y+C3.s0*gam.z;
    	outvA.y =  C1.s1*gam.x+C2.s1*gam.y+C3.s1*gam.z;
    	outvA.z =  C1.s2*gam.x+C2.s2*gam.y+C3.s2*gam.z;
    	outvA.w =  C1.s3*gam.x+C2.s3*gam.y+C3.s3*gam.z;

    	outoA.x =  C1.s4*gam.x+C2.s4*gam.y+C3.s4*gam.z;
    	outoA.y =  C1.s5*gam.x+C2.s5*gam.y+C3.s5*gam.z;
    	outoA.z =  C1.s6*gam.x+C2.s6*gam.y+C3.s6*gam.z;
    	outoA.w =  C1.s7*gam.x+C2.s7*gam.y+C3.s7*gam.z;


    	outvB.x =  C1.s8 *gam.x+C2.s8 *gam.y+C3.s8 *gam.z;
    	outvB.y =  C1.s9 *gam.x+C2.s9 *gam.y+C3.s9 *gam.z;
    	outvB.z =  C1.s10*gam.x+C2.s10*gam.y+C3.s10*gam.z;
    	outvB.w =  C1.s11*gam.x+C2.s11*gam.y+C3.s11*gam.z;

    	outoB.x =  C1.s12*gam.x+C2.s12*gam.y+C3.s12*gam.z;
    	outoB.y =  C1.s13*gam.x+C2.s13*gam.y+C3.s13*gam.z;
    	outoB.z =  C1.s14*gam.x+C2.s14*gam.y+C3.s14*gam.z;
    	outoB.w =  C1.s15*gam.x+C2.s15*gam.y+C3.s15*gam.z;


    	out_velocityA[id]= outvA;
    	out_omegaA[id]= outoA;

    	out_velocityB[id]= outvB;
    	out_omegaB[id]= outoB;
    }
}


