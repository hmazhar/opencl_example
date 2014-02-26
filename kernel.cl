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
	__global float3 *Jxyz, 
	__global float3 *Juvw, 
	__global float3 *gamma, 
	__global float3 *out_velocity, 
	__global float3 *out_omega, 
	const unsigned int n_contact)
{
    const unsigned int id = get_global_id(0);
    if (id < n_contact){
    	float3 gam = gamma[id];

    	out_velocity[id]= Jxyz[id+n_contact*0]*gam.x+Jxyz[id+n_contact*1]*gam.y+Jxyz[id+n_contact*2]*gam.z;
    	out_omega[id]= Juvw[id+n_contact*0]*gam.x+Juvw[id+n_contact*1]*gam.y+Juvw[id+n_contact*2]*gam.z;
    }
}


