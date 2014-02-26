#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecMathA(__global double *a, __global double *b, __global double *c, const unsigned int n)
{
    const unsigned int id = get_global_id(0);
    if (id < n){
    	double _a_ = a[id];
    	double _b_ = b[id];
		for(int i=0; i<100; i++){
			double _a = _a_ + _b_ * _b_ +(_a_ * _b_)/_b_;
			double _b = _a_ - _b_ / _b_ -(_a_ / _b_)*_b_;
			double _c = _b_ + _a_ * _a_ +(_b_ * _a_)/_b_;
			double _d = _b_ - _a_ / _b_ -(_b_ / _b_)*_a_;
			c[id] = _a + _b * _c +(_a * _d)/_c;
		}
    }
}


#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecMathB(__global double *a, __global double *b, __global double *c, const unsigned int n)
{
    const unsigned int id = get_global_id(0);
    if (id < n){
		for(int i=0; i<100; i++){
			double _a = a[id] + b[id] * b[id] +(a[id] * b[id])/b[id];
			double _b = a[id] - b[id] / b[id] -(a[id] / b[id])*b[id];
			double _c = b[id] + a[id] * a[id] +(b[id] * a[id])/b[id];
			double _d = b[id] - a[id] / b[id] -(b[id] / b[id])*a[id];
			c[id] = _a + _b * _c +(_a * _d)/_c;
		}
    }
}


#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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

    	out_velocityA[id]= JxyzA[id+n_contact*0]*gam.x+JxyzA[id+n_contact*1]*gam.y+JxyzA[id+n_contact*2]*gam.z;
    	out_omegaA[id]= JuvwA[id+n_contact*0]*gam.x+JuvwA[id+n_contact*1]*gam.y+JuvwA[id+n_contact*2]*gam.z;

    	out_velocityB[id]= JxyzB[id+n_contact*0]*gam.x+JxyzB[id+n_contact*1]*gam.y+JxyzB[id+n_contact*2]*gam.z;
    	out_omegaB[id]= JuvwB[id+n_contact*0]*gam.x+JuvwB[id+n_contact*1]*gam.y+JuvwB[id+n_contact*2]*gam.z;
    }
}


