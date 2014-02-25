#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecMathA(__global double *a, __global double *b, __global double *c, const unsigned int n)
{
    int id = get_global_id(0);
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
    int id = get_global_id(0);
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
	__global double *Jx, 
	__global double *Jy, 
	__global double *Jz, 
	__global double *Ju, 
	__global double *Jv, 
	__global double *Jw, 
	__global double *gamma_x, 
	__global double *gamma_y, 
	__global double *gamma_z,
	__global double *out_velocity_x, 
	__global double *out_velocity_y, 
	__global double *out_velocity_z,
	__global double *out_omega_x, 
	__global double *out_omega_y, 
	__global double *out_omega_z,
	const unsigned int n_contact)
{
    int id = get_global_id(0);
    if (id < n_contact){
    	double gam_x = gamma_x[id];
    	double gam_y = gamma_y[id];
    	double gam_z = gamma_z[id];

    	out_velocity_x[id]= Jx[id+n_contact*0]*gam_x+Jx[id+n_contact*1]*gam_y+Jx[id+n_contact*2]*gam_z;
    	out_velocity_y[id]= Jy[id+n_contact*0]*gam_x+Jy[id+n_contact*1]*gam_y+Jy[id+n_contact*2]*gam_z;
    	out_velocity_z[id]= Jz[id+n_contact*0]*gam_x+Jz[id+n_contact*1]*gam_y+Jz[id+n_contact*2]*gam_z;

    	out_omega_x[id]= Ju[id+n_contact*0]*gam_x+Ju[id+n_contact*1]*gam_y+Ju[id+n_contact*2]*gam_z;
    	out_omega_y[id]= Jv[id+n_contact*0]*gam_x+Jv[id+n_contact*1]*gam_y+Jv[id+n_contact*2]*gam_z;
    	out_omega_z[id]= Jw[id+n_contact*0]*gam_x+Jw[id+n_contact*1]*gam_y+Jw[id+n_contact*2]*gam_z;
    }
}


