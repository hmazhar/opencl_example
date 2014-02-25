#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecAdd(__global double *a, __global double *b, __global double *c, const unsigned int n)
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