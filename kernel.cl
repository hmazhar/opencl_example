#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecAdd(__global double *a, __global double *b, __global double *c, const unsigned int n)
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