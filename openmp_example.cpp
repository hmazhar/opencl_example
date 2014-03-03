#include "omp.h"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <nmmintrin.h>
using namespace std;
static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

class __attribute__ ((aligned(16))) float4 {
public:
	union {
		struct {
			float x, y, z, w;
		};
		__m128 mmvalue;
	};

	inline float4() :
			mmvalue(_mm_setzero_ps()) {
	}
	inline float4(float a) :
			mmvalue(_mm_set1_ps(a)) {
	}
	inline float4(float a, float b, float c) :
			mmvalue(_mm_setr_ps(0, a, b, c)) {
	}
	inline float4(float a, float b, float c, float d) :
			mmvalue(_mm_setr_ps(a, b, c, d)) {
	}
	inline float4(__m128 m) :
			mmvalue(m) {
	}

	inline float4 operator+(const float4& b) const {
		return _mm_add_ps(mmvalue, b.mmvalue);
	}
	inline float4 operator-(const float4& b) const {
		return _mm_sub_ps(mmvalue, b.mmvalue);
	}
	inline float4 operator*(const float4& b) const {
		return _mm_mul_ps(mmvalue, b.mmvalue);
	}
	inline float4 operator/(const float4& b) const {
		return _mm_div_ps(mmvalue, b.mmvalue);
	}
	inline float4 operator-() const {
		return _mm_xor_ps(mmvalue, SIGNMASK);
	}

	inline float4& operator+=(const float4& b) {
		*this = *this + b;
		return *this;
	}
	inline float4& operator-=(const float4& b) {
		*this = *this - b;
		return *this;
	}
	inline float4& operator*=(const float4& b) {
		*this = *this * b;
		return *this;
	}
	inline float4& operator/=(const float4& b) {
		*this = *this / b;
		return *this;
	}

	inline float4 operator+(float b) const {
		return _mm_add_ps(mmvalue, _mm_set1_ps(b));
	}
	inline float4 operator-(float b) const {
		return _mm_sub_ps(mmvalue, _mm_set1_ps(b));
	}
	inline float4 operator*(float b) const {
		return _mm_mul_ps(mmvalue, _mm_set1_ps(b));
	}
	inline float4 operator/(float b) const {
		return _mm_div_ps(mmvalue, _mm_set1_ps(b));
	}

	inline float4& operator+=(float b) {
		*this = *this + b;
		return *this;
	}
	inline float4& operator-=(float b) {
		*this = *this - b;
		return *this;
	}
	inline float4& operator*=(float b) {
		*this = *this * b;
		return *this;
	}
	inline float4& operator/=(float b) {
		*this = *this / b;
		return *this;
	}

	inline float length() const {
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(mmvalue, mmvalue, 0x71)));
	}
	inline float rlength() const {
		return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_dp_ps(mmvalue, mmvalue, 0x71)));
	}

	inline float4 cross(const float4 &b) const {
		return _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mmvalue, mmvalue, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b.mmvalue, b.mmvalue, _MM_SHUFFLE(3, 1, 0, 2))),
				_mm_mul_ps(_mm_shuffle_ps(mmvalue, mmvalue, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b.mmvalue, b.mmvalue, _MM_SHUFFLE(3, 0, 2, 1))));
	}
};

inline float4 operator+(float a, const float4& b) {
	return b + a;
}
inline float4 operator-(float a, const float4& b) {
	return float4(a) - b;
}
inline float4 operator*(float a, const float4& b) {
	return b * a;
}
inline float4 operator/(float a, const float4& b) {
	return float4(a) / b;
}
inline float length(const float4& a) {
	return a.length();
}
inline float rlength(const float4& a) {
	return a.rlength();
}

inline float4 cross(const float4& a, const float4& b) {
	return a.cross(b);
}

void F(int n_contact, float* h_g, float4* JxA, float4* JuA, float4* JuB, float4* out_vel_A, float4* out_omg_A) {

#pragma omp parallel for
	for (int id = 0; id < n_contact; id++) {

		float gam = h_g[id];

		float4 U = JxA[id];

		out_vel_A[id] = -U * gam;
		out_vel_A[id + n_contact] = U * gam;
		out_omg_A[id] = JuA[id] * gam;
		out_omg_A[id + n_contact] = JuB[id] * gam;

	}
}

int main(int argc, char *argv[]) {

	int thread_num = 1;
	if (argc > 1) {

		thread_num = atoi(argv[1]);
	}
	unsigned int contacts = 1024000 / 2;
	if (argc > 2) {
		contacts = atoi(argv[2]);
	}

	omp_set_num_threads(thread_num);
	// Length of vectors

	unsigned int constraints = contacts * 3;

	// Host input vectors
	float *gamma_x, *gamma_y, *gamma_z;
// Host output vectors
	float *out_vel_x, *out_vel_y, *out_vel_z;
	float *out_omg_x, *out_omg_y, *out_omg_z;

	// Size, in bytes, of each vector
	size_t bytes = contacts * sizeof(float);

	// Allocate memory for each vector on host
	float4 * JxA = (float4*) malloc(contacts * sizeof(float4));
	float4 * JyA = (float4*) malloc(contacts * sizeof(float4));
	float4 * JzA = (float4*) malloc(contacts * sizeof(float4));

	float4 * JuA = (float4*) malloc(contacts * sizeof(float4));
	float4 * JvA = (float4*) malloc(contacts * sizeof(float4));
	float4 * JwA = (float4*) malloc(contacts * sizeof(float4));

	float4 * JxB = (float4*) malloc(contacts * sizeof(float4));
	float4 * JyB = (float4*) malloc(contacts * sizeof(float4));
	float4 * JzB = (float4*) malloc(contacts * sizeof(float4));

	float4 * JuB = (float4*) malloc(contacts * sizeof(float4));
	float4 * JvB = (float4*) malloc(contacts * sizeof(float4));
	float4 * JwB = (float4*) malloc(contacts * sizeof(float4));

	float * h_g = (float*) malloc(contacts*3 * sizeof(float));

	float4 * out_vel_A = (float4*) malloc(contacts*2 * sizeof(float4));
	float4 * out_omg_A = (float4*) malloc(contacts*2 * sizeof(float4));
	float4 * out_vel_B = (float4*) malloc(contacts * sizeof(float4));
	float4 * out_omg_B = (float4*) malloc(contacts * sizeof(float4));

	// Initialize vectors on host
	int i;
	for (i = 0; i < contacts; i++) {
		JxA[i].x = sinf(i) * sinf(i);
		JxA[i].y = sinf(i) * sinf(i);
		JxA[i].z = sinf(i) * sinf(i);

		JyA[i].x = sinf(i) * sinf(i);
		JyA[i].y = sinf(i) * sinf(i);
		JyA[i].z = sinf(i) * sinf(i);

		JzA[i].x = sinf(i) * sinf(i);
		JzA[i].y = sinf(i) * sinf(i);
		JzA[i].z = sinf(i) * sinf(i);

		h_g[i] = sinf(i) * sinf(i);
		h_g[i+contacts] = sinf(i) * sinf(i);
		h_g[i+contacts*2] = sinf(i) * sinf(i);

		JuA[i].x = sinf(i) * sinf(i);
		JuA[i].y = sinf(i) * sinf(i);
		JuA[i].z = sinf(i) * sinf(i);

		JvA[i].x = cosf(i) * cosf(i);
		JvA[i].y = cosf(i) * cosf(i);
		JvA[i].z = cosf(i) * cosf(i);

		JwA[i].x = cosf(i) * cosf(i);
		JwA[i].y = cosf(i) * cosf(i);
		JwA[i].z = cosf(i) * cosf(i);

		JxB[i].x = sinf(i) * sinf(i);
		JxB[i].y = sinf(i) * sinf(i);
		JxB[i].z = sinf(i) * sinf(i);

		JyB[i].x = cosf(i) * cosf(i);
		JyB[i].y = cosf(i) * cosf(i);
		JyB[i].z = cosf(i) * cosf(i);

		JzB[i].x = cosf(i) * cosf(i);
		JzB[i].y = cosf(i) * cosf(i);
		JzB[i].z = cosf(i) * cosf(i);

		JuB[i].x = sinf(i) * sinf(i);
		JuB[i].y = sinf(i) * sinf(i);
		JuB[i].z = sinf(i) * sinf(i);

		JvB[i].x = cosf(i) * cosf(i);
		JvB[i].y = cosf(i) * cosf(i);
		JvB[i].z = cosf(i) * cosf(i);

		JwB[i].x = cosf(i) * cosf(i);
		JwB[i].y = cosf(i) * cosf(i);
		JwB[i].z = cosf(i) * cosf(i);

	}

	int n_contact = contacts;

	double total_time_omp;
	double total_flops;
	double total_memory;
	double runs = 10;
	for (int i = 0; i < runs; i++) {

		double start = omp_get_wtime();


		F(contacts,  h_g, JxA, JuA,  JuB,  out_vel_A, out_omg_A);


		double end = omp_get_wtime();

		total_time_omp = (end - start) * 1000;
		total_flops = 12 * contacts / ((end - start)) / 1e9;
		total_memory = (7 * 4 * 4 + 1 * 4) * contacts / ((end - start)) / 1024.0 / 1024.0 / 1024.0;
		printf("\nExecution time in milliseconds =  %0.3f ms | %0.3f Gflop | %0.3f GB/s \n", total_time_omp , total_flops , total_memory );
	}



	//release host memory
	free(JxA);
	free(JyA);
	free(JzA);

	free(JuA);
	free(JvA);
	free(JwA);

	free(JxB);
	free(JyB);
	free(JzB);

	free(JuB);
	free(JvB);
	free(JwB);

	free(h_g);

	free(out_vel_A);
	free(out_omg_A);
	free(out_vel_B);
	free(out_omg_B);

	return 0;
}
