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
	unsigned int contacts = 1024000/2;
	unsigned int constraints = contacts*3;

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
	size_t bytes = constraints * sizeof(cl_float3);

	// Allocate memory for each vector on host
	cl_float3 * h_jxyzA = (cl_float3*) malloc(bytes);
	cl_float3 * h_juvwA = (cl_float3*) malloc(bytes);
	cl_float3 * h_jxyzB = (cl_float3*) malloc(bytes);
	cl_float3 * h_juvwB = (cl_float3*) malloc(bytes);
	cl_float3 * h_g = (cl_float3*) malloc(bytes);
	cl_float3 * h_vA = (cl_float3*) malloc(bytes);
	cl_float3 * h_oA = (cl_float3*) malloc(bytes);
	cl_float3 * h_vB = (cl_float3*) malloc(bytes);
	cl_float3 * h_oB = (cl_float3*) malloc(bytes);

	// Initialize vectors on host
	int i;
	for (i = 0; i < constraints; i++) {
		h_jxyzA[i].s[0] = sinf(i) * sinf(i);
		h_jxyzA[i].s[1] = sinf(i) * sinf(i);
		h_jxyzA[i].s[2] = sinf(i) * sinf(i);

		h_juvwA[i].s[0] = sinf(i) * sinf(i);
		h_juvwA[i].s[1] = sinf(i) * sinf(i);
		h_juvwA[i].s[2] = sinf(i) * sinf(i);

		h_jxyzB[i].s[0] = sinf(i) * sinf(i);
		h_jxyzB[i].s[1] = sinf(i) * sinf(i);
		h_jxyzB[i].s[2] = sinf(i) * sinf(i);

		h_juvwB[i].s[0] = sinf(i) * sinf(i);
		h_juvwB[i].s[1] = sinf(i) * sinf(i);
		h_juvwB[i].s[2] = sinf(i) * sinf(i);

		h_g[i].s[0] = sinf(i) * sinf(i);
		h_g[i].s[1] = sinf(i) * sinf(i);
		h_g[i].s[2] = sinf(i) * sinf(i);
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
	clBuildProgram(program, 1, &deviceIds[device_num], "-cl-mad-enable", NULL, NULL);
	size_t len = 0;
	clGetProgramBuildInfo(program, deviceIds[device_num], CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
	char *log = new char[len]; //or whatever you use
	clGetProgramBuildInfo(program, deviceIds[device_num], CL_PROGRAM_BUILD_LOG, len, log, NULL);
	printf("Build Log:\n%s\n", log);
	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "ShurA", &err);


	// Create the input and output arrays in device memory for our calculation
	cl_mem d_jxyzA = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jxyzA, NULL);
	cl_mem d_juvwA = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_juvwA, NULL);

	cl_mem d_jxyzB = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_jxyzB, NULL);
	cl_mem d_juvwB = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_juvwB, NULL);

	cl_mem d_g = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_g, NULL);
	cl_mem d_vA = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_vA, NULL);
	cl_mem d_oA = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_oA, NULL);

	cl_mem d_vB = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_vB, NULL);
	cl_mem d_oB = clCreateBuffer(context,  CL_MEM_USE_HOST_PTR , bytes, h_oB, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_jxyzA, CL_TRUE, 0, bytes, h_jxyzA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_juvwA, CL_TRUE, 0, bytes, h_juvwA, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_jxyzB, CL_TRUE, 0, bytes, h_jxyzB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_juvwB, CL_TRUE, 0, bytes, h_juvwB, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_g, CL_TRUE, 0, bytes, h_g, 0, NULL, NULL);



	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_jxyzA);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_juvwA);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_jxyzB);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_juvwB);
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_g);
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_vA);
	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_oA);
	err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_vB);
	err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_oB);
	err = clSetKernelArg(kernel, 9, sizeof(unsigned int), &contacts);

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
	clReleaseMemObject(d_jxyzA);
	clReleaseMemObject(d_juvwA);
	clReleaseMemObject(d_jxyzB);
	clReleaseMemObject(d_juvwB);
	clReleaseMemObject(d_g);
	clReleaseMemObject(d_vA);
	clReleaseMemObject(d_oA);
	clReleaseMemObject(d_vB);
	clReleaseMemObject(d_oB);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(h_jxyzA);
	free(h_juvwA);
	free(h_jxyzB);
	free(h_juvwB);
	free(h_g);
	free(h_vA);
	free(h_oA);
	free(h_vB);
	free(h_oB);
	return 0;
}
