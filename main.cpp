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
	unsigned int contacts = 1024000*3;
	unsigned int constraints = contacts*3;

	double *h_jx, *h_jy, *h_jz;
	double *h_ju, *h_jv, *h_jw;
	double *h_gx, *h_gy, *h_gz;

	double *h_vx, *h_vy, *h_vz;
	double *h_ox, *h_oy, *h_oz;

	cl_mem d_jx, d_jy, d_jz;
	cl_mem d_ju, d_jv, d_jw;
	cl_mem d_gx, d_gy, d_gz;

	cl_mem d_vx, d_vy, d_vz;
	cl_mem d_ox, d_oy, d_oz;

	std::vector<cl_platform_id> platformIds;     // OpenCL platform
	std::vector<cl_device_id> deviceIds;         // device ID
	cl_context context;                          // context
	cl_command_queue queue;                      // command queue
	cl_program program;                          // program
	cl_kernel kernel;                            // kernel
	cl_event prof_event;
	cl_uint deviceIdCount = 0;

	int device_num = 0;
	if (argc > 1) {
		device_num = atoi(argv[1]);
	}

	// Size, in bytes, of each vector
	size_t bytes = constraints * sizeof(double);

	// Allocate memory for each vector on host
	h_jx = (double*) malloc(bytes);
	h_jy = (double*) malloc(bytes);
	h_jz = (double*) malloc(bytes);

	h_ju = (double*) malloc(bytes);
	h_jv = (double*) malloc(bytes);
	h_jw = (double*) malloc(bytes);

	h_gx = (double*) malloc(bytes);
	h_gy = (double*) malloc(bytes);
	h_gz = (double*) malloc(bytes);

	h_vx = (double*) malloc(bytes);
	h_vy = (double*) malloc(bytes);
	h_vz = (double*) malloc(bytes);

	h_ox = (double*) malloc(bytes);
	h_oy = (double*) malloc(bytes);
	h_oz = (double*) malloc(bytes);

	// Initialize vectors on host
	int i;
	for (i = 0; i < constraints; i++) {
		h_jx[i] = sinf(i) * sinf(i);
		h_jy[i] = cosf(i) * cosf(i);
		h_jz[i] = cosf(i) * cosf(i);

		h_ju[i] = sinf(i) * sinf(i);
		h_jv[i] = cosf(i) * cosf(i);
		h_jw[i] = cosf(i) * cosf(i);

		h_gx[i] = sinf(i) * sinf(i);
		h_gy[i] = cosf(i) * cosf(i);
		h_gz[i] = cosf(i) * cosf(i);

	}

	size_t globalSize, localSize;
	cl_int err;

	// Number of work items in each local work group
	localSize = 128;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(contacts / (float) localSize) * localSize;
	// Bind to platform
	platformIds = GetPlatforms();
	// Get ID for the device
	deviceIds = GetDevices(platformIds[0]);

	// Create a context
	context = clCreateContext(0, 1, &deviceIds[device_num], NULL, NULL, &err);

	// Create a command queue
	queue = clCreateCommandQueue(context, deviceIds[device_num], CL_QUEUE_PROFILING_ENABLE, &err);

	// Create the compute program from the source buffer
	program = CreateProgram(LoadKernel("kernel.cl"), context);

	// Build the program executable
	clBuildProgram(program, 0, NULL, "-cl-mad-enable -cl-denorms-are-zero", NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "ShurA", &err);


	// Create the input and output arrays in device memory for our calculation
	d_jx = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jx, NULL);
	d_jy = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jy, NULL);
	d_jz = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jz, NULL);

	d_ju = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_ju, NULL);
	d_jv = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jv, NULL);
	d_jw = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jw, NULL);

	d_gx = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_gx, NULL);
	d_gy = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_gy, NULL);
	d_gz = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_gz, NULL);

	d_vx = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_vx, NULL);
	d_vy = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_vy, NULL);
	d_vz = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_vz, NULL);

	d_ox = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_ox, NULL);
	d_oy = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_oy, NULL);
	d_oz = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_oz, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_jx, CL_TRUE, 0, bytes, h_jx, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jy, CL_TRUE, 0, bytes, h_jy, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jz, CL_TRUE, 0, bytes, h_jz, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_ju, CL_TRUE, 0, bytes, h_ju, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jv, CL_TRUE, 0, bytes, h_jv, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jw, CL_TRUE, 0, bytes, h_jw, 0, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_gx, CL_TRUE, 0, bytes, h_gx, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_gy, CL_TRUE, 0, bytes, h_gy, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_gz, CL_TRUE, 0, bytes, h_gz, 0, NULL, NULL);




	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_jx);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_jy);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_jz);

	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_ju);
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_jv);
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_jw);

	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_gx);
	err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_gy);
	err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_gz);

	err = clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_vx);
	err = clSetKernelArg(kernel, 10, sizeof(cl_mem), &d_vy);
	err = clSetKernelArg(kernel, 11, sizeof(cl_mem), &d_vz);

	err = clSetKernelArg(kernel, 12, sizeof(cl_mem), &d_ox);
	err = clSetKernelArg(kernel, 13, sizeof(cl_mem), &d_oy);
	err = clSetKernelArg(kernel, 14, sizeof(cl_mem), &d_oz);
	err = clSetKernelArg(kernel, 15, sizeof(unsigned int), &contacts);

	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	double start = omp_get_wtime();
	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &prof_event);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	double end = omp_get_wtime();
	clWaitForEvents(1, &prof_event);
	// Read the results from the device
	//clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

	//Sum up vector c and print result divided by n, this should equal 1 within error
	// double sum = 0;
	// for (i = 0; i < n; i++) {
	// 	sum += h_c[i];
	// }
	// printf("final result: %f\n", sum / n);
	cl_ulong time_start, time_end;
	double total_time;
	clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time in milliseconds = %0.3f ms |  %0.3f ms\n", (total_time / 1000000.0), (end - start) * 1000);

	// release OpenCL resources
	clReleaseMemObject(d_jx);
	clReleaseMemObject(d_jy);
	clReleaseMemObject(d_jz);

	clReleaseMemObject(d_ju);
	clReleaseMemObject(d_jv);
	clReleaseMemObject(d_jw);

	clReleaseMemObject(d_gx);
	clReleaseMemObject(d_gy);
	clReleaseMemObject(d_gz);

	clReleaseMemObject(d_vx);
	clReleaseMemObject(d_vy);
	clReleaseMemObject(d_vz);

	clReleaseMemObject(d_ox);
	clReleaseMemObject(d_oy);
	clReleaseMemObject(d_oz);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(h_jx);
	free(h_jy);
	free(h_jz);

	free(h_ju);
	free(h_jv);
	free(h_jw);

	free(h_gx);
	free(h_gy);
	free(h_gz);

	free(h_vx);
	free(h_vy);
	free(h_vz);

	free(h_ox);
	free(h_oy);
	free(h_oz);
	return 0;
}
