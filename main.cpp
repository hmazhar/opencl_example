#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif
#include "omp.h"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <time.h>
using namespace std;

#define CREATE_BUFFER(context, x,y, size) x=clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , size, y, NULL);


std::string GetPlatformName(cl_platform_id id) {
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*>(result.data()), nullptr);
	return result;
}

std::string GetDeviceName(cl_device_id id) {
	size_t size = 0;
	size_t wg_size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*>(result.data()), nullptr);
	return result;
}

void CheckError(cl_int error) {
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit(1);
	}
}

std::string LoadKernel(const char* name) {
	std::ifstream in(name);
	std::string result((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	return result;
}

cl_program CreateProgram(const std::string& source, cl_context context) {
	size_t lengths[1] = { source.size() };
	const char* sources[1] = { source.data() };
	cl_int error = 0;
	cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &error);
	CheckError(error);
	return program;
}
std::vector<cl_platform_id> GetPlatforms() {
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		exit(1);
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}
	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
	}
	return platformIds;
}

std::vector<cl_device_id> GetDevices(cl_platform_id platform) {
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		exit(1);
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;
	}
	return deviceIds;
}

int main(int argc, char *argv[]) {




	std::vector<cl_platform_id> platformIds;     // OpenCL platform
	std::vector<cl_device_id> deviceIds;         // device ID
	cl_event prof_event;
	cl_uint deviceIdCount = 0;

	size_t globalSize, localSize;
	cl_int err;

	int device_num = 0;
	if (argc > 1) {
		device_num = atoi(argv[1]);
	}
	unsigned int contacts = 102400;
	if (argc > 2) {
		contacts = atoi(argv[2]);
	}

	
	unsigned int constraints = contacts*3;

	// Size, in bytes, of each vector
	size_t bytes = constraints * sizeof(float);

	// Allocate memory for each vector on host
	float * h_jxA = (float*) malloc(bytes);
	float * h_jyA = (float*) malloc(bytes);
	float * h_jzA = (float*) malloc(bytes);

	float * h_juA = (float*) malloc(bytes);
	float * h_jvA = (float*) malloc(bytes);
	float * h_jwA = (float*) malloc(bytes);

	float * h_jxB = (float*) malloc(bytes);
	float * h_jyB = (float*) malloc(bytes);
	float * h_jzB = (float*) malloc(bytes);

	float * h_juB = (float*) malloc(bytes);
	float * h_jvB = (float*) malloc(bytes);
	float * h_jwB = (float*) malloc(bytes);



	float * h_gx = (float*) malloc(bytes);
	float * h_gy = (float*) malloc(bytes);
	float * h_gz = (float*) malloc(bytes);

	float * h_vxA = (float*) malloc(bytes);
	float * h_vyA = (float*) malloc(bytes);
	float * h_vzA = (float*) malloc(bytes);

	float * h_oxA = (float*) malloc(bytes);
	float * h_oyA = (float*) malloc(bytes);
	float * h_ozA = (float*) malloc(bytes);

	float * h_vxB = (float*) malloc(bytes);
	float * h_vyB = (float*) malloc(bytes);
	float * h_vzB = (float*) malloc(bytes);

	float * h_oxB = (float*) malloc(bytes);
	float * h_oyB = (float*) malloc(bytes);
	float * h_ozB = (float*) malloc(bytes);

	// Initialize vectors on host
	int i;
	for (i = 0; i < constraints; i++) {
		h_jxA[i] = sinf(i) * sinf(i);
		h_jyA[i] = cosf(i) * cosf(i);
		h_jzA[i] = cosf(i) * cosf(i);

		h_juA[i] = sinf(i) * sinf(i);
		h_jvA[i] = cosf(i) * cosf(i);
		h_jwA[i] = cosf(i) * cosf(i);

		h_gx[i] = sinf(i) * sinf(i);
		h_gy[i] = cosf(i) * cosf(i);
		h_gz[i] = cosf(i) * cosf(i);

		h_jxB[i] = sinf(i) * sinf(i);
		h_jyB[i] = cosf(i) * cosf(i);
		h_jzB[i] = cosf(i) * cosf(i);

		h_juB[i] = sinf(i) * sinf(i);
		h_jvB[i] = cosf(i) * cosf(i);
		h_jwB[i] = cosf(i) * cosf(i);


	}

	

	// Number of work items in each local work group
	localSize = 128;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(contacts / (float) localSize) * localSize;
	// Bind to platform
	platformIds = GetPlatforms();
	// Get ID for the device
	deviceIds = GetDevices(platformIds[0]);

	// Create a context
	cl_context context = clCreateContext(0, 1, &deviceIds[device_num], NULL, NULL, &err);

	// Create a command queue
	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[device_num], CL_QUEUE_PROFILING_ENABLE, &err);

	// Create the compute program from the source buffer
	cl_program program = CreateProgram(LoadKernel("kernel.cl"), context);

	// Build the program executable
	clBuildProgram(program, 0, NULL, "-cl-mad-enable", NULL, NULL);

	// Create the compute kernel in the program we wish to run
	cl_kernel kernel = clCreateKernel(program, "KERNEL_1_0", &err);


	// Create the input and output arrays in device memory for our calculation
	cl_mem d_jxA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jyA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jzA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);

	cl_mem d_juA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jvA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jwA = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);

	cl_mem d_jxB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jyB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jzB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);

	cl_mem d_juB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jvB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_jwB = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);


	cl_mem d_gx = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_gy = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);
	cl_mem d_gz = clCreateBuffer(context,  CL_MEM_READ_ONLY , bytes, NULL, NULL);

	cl_mem d_vxA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_vyA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_vzA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);

	cl_mem d_oxA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_oyA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_ozA = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);

	cl_mem d_vxB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_vyB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_vzB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);

	cl_mem d_oxB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_oyB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);
	cl_mem d_ozB = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , bytes,NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_jxA, CL_TRUE, 0, bytes, h_jxA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jyA, CL_TRUE, 0, bytes, h_jyA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jzA, CL_TRUE, 0, bytes, h_jzA, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_juA, CL_TRUE, 0, bytes, h_juA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jvA, CL_TRUE, 0, bytes, h_jvA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jwA, CL_TRUE, 0, bytes, h_jwA, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_jxB, CL_TRUE, 0, bytes, h_jxB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jyB, CL_TRUE, 0, bytes, h_jyB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jzB, CL_TRUE, 0, bytes, h_jzB, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_juB, CL_TRUE, 0, bytes, h_juB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jvB, CL_TRUE, 0, bytes, h_jvB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jwB, CL_TRUE, 0, bytes, h_jwB, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_gx, CL_TRUE, 0, bytes, h_gx, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_gy, CL_TRUE, 0, bytes, h_gy, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_gz, CL_TRUE, 0, bytes, h_gz, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_jxA);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_jyA);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_jzA);

	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_juA);
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_jvA);
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_jwA);

	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_jxB);
	err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_jyB);
	err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_jzB);

	err = clSetKernelArg(kernel, 9,  sizeof(cl_mem), &d_juB);
	err = clSetKernelArg(kernel, 10, sizeof(cl_mem), &d_jvB);
	err = clSetKernelArg(kernel, 11, sizeof(cl_mem), &d_jwB);

	err = clSetKernelArg(kernel, 12, sizeof(cl_mem), &d_gx);
	err = clSetKernelArg(kernel, 13, sizeof(cl_mem), &d_gy);
	err = clSetKernelArg(kernel, 14, sizeof(cl_mem), &d_gz);

	err = clSetKernelArg(kernel, 15, sizeof(cl_mem), &d_vxA);
	err = clSetKernelArg(kernel, 16, sizeof(cl_mem), &d_vyA);
	err = clSetKernelArg(kernel, 17, sizeof(cl_mem), &d_vzA);

	err = clSetKernelArg(kernel, 18, sizeof(cl_mem), &d_oxA);
	err = clSetKernelArg(kernel, 19, sizeof(cl_mem), &d_oyA);
	err = clSetKernelArg(kernel, 20, sizeof(cl_mem), &d_ozA);

	err = clSetKernelArg(kernel, 21, sizeof(cl_mem), &d_vxB);
	err = clSetKernelArg(kernel, 22, sizeof(cl_mem), &d_vyB);
	err = clSetKernelArg(kernel, 23, sizeof(cl_mem), &d_vzB);

	err = clSetKernelArg(kernel, 24, sizeof(cl_mem), &d_oxB);
	err = clSetKernelArg(kernel, 25, sizeof(cl_mem), &d_oyB);
	err = clSetKernelArg(kernel, 26, sizeof(cl_mem), &d_ozB);
	err = clSetKernelArg(kernel, 27, sizeof(unsigned int), &contacts);

	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);
double total_time_cl;
double total_time_omp;
double total_flops;
double total_memory;
double  runs = 100;
for(int i=0; i<runs; i++){

	double start = omp_get_wtime();
	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &prof_event);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	double end = omp_get_wtime();
	clWaitForEvents(1, &prof_event);
	// Read the results from the device
	//clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);


	cl_ulong time_start, time_end;
	
	clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time_cl += (time_end - time_start)/ 1000000.0;
	total_time_omp += (end - start) * 1000;
	total_flops += 60*contacts/((time_end - time_start)/ 1000000000.0)/1e9;
	total_memory+= 204*contacts/((time_end - time_start)/ 1000000000.0)/1024.0/1024.0/1024.0;
}


	printf("\nExecution time in milliseconds = %0.3f ms |  %0.3f ms | %0.3f Gflop | %0.3f GB/s \n", (total_time_cl / runs), total_time_omp/runs, total_flops/runs, total_memory/runs);

	// release OpenCL resources
	clReleaseMemObject(d_jxA);
	clReleaseMemObject(d_jyA);
	clReleaseMemObject(d_jzA);

	clReleaseMemObject(d_juA);
	clReleaseMemObject(d_jvA);
	clReleaseMemObject(d_jwA);

	clReleaseMemObject(d_jxB);
	clReleaseMemObject(d_jyB);
	clReleaseMemObject(d_jzB);

	clReleaseMemObject(d_juB);
	clReleaseMemObject(d_jvB);
	clReleaseMemObject(d_jwB);

	clReleaseMemObject(d_gx);
	clReleaseMemObject(d_gy);
	clReleaseMemObject(d_gz);


	clReleaseMemObject(d_vxA);
	clReleaseMemObject(d_vyA);
	clReleaseMemObject(d_vzA);

	clReleaseMemObject(d_oxA);
	clReleaseMemObject(d_oyA);
	clReleaseMemObject(d_ozA);

	clReleaseMemObject(d_vxB);
	clReleaseMemObject(d_vyB);
	clReleaseMemObject(d_vzB);

	clReleaseMemObject(d_oxB);
	clReleaseMemObject(d_oyB);
	clReleaseMemObject(d_ozB);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(h_jxA);
	free(h_jyA);
	free(h_jzA);

	free(h_juA);
	free(h_jvA);
	free(h_jwA);

	free(h_jxB);
	free(h_jyB);
	free(h_jzB);

	free(h_juB);
	free(h_jvB);
	free(h_jwB);

	free(h_gx);
	free(h_gy);
	free(h_gz);

	free(h_vxA);
	free(h_vyA);
	free(h_vzA);

	free(h_oxA);
	free(h_oyA);
	free(h_ozA);

	free(h_vxB);
	free(h_vyB);
	free(h_vzB);

	free(h_oxB);
	free(h_oyB);
	free(h_ozB);
	return 0;
}
