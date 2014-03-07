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

void Function_1(int n_contact, float4* h_g, float4* norm, float4* JuA, float4* JuB, float4* JvA, float4* JvB, float4* JwA, float4* JwB, float4* out_vel_A, float4* out_omg_A,
		float4* out_vel_B, float4* out_omg_B) {

#pragma omp parallel for
	for (int id = 0; id < n_contact; id++) {

		float4 gam = h_g[id];

		float4 U = norm[id], V, W;

		W = cross(U, (float4) (0, 1, 0));
		float mzlen = length(W);

		if (mzlen < 0.0001f) {
			float4 mVsingular = (float4) (1, 0, 0);
			W = cross(U, mVsingular);
			mzlen = length(W);
		}
		W = W * 1.0f / mzlen;
		V = cross(W, U);

		out_vel_A[id] = -U * gam - V * gam - W * gam;
		out_vel_B[id] = U * gam + V * gam + W * gam;
		out_omg_A[id] = JuA[id] * gam + JvA[id] * gam + JwA[id] * gam;
		out_omg_B[id] = JuB[id] * gam + JvB[id] * gam + JwB[id] * gam;

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

	// Size, in bytes, of each vector
	size_t bytes = contacts * sizeof(float);

	// Allocate memory for each vector on host

	vector<float4> JxA(contacts);
	vector<float4> JyA(contacts);
	vector<float4> JzA(contacts);

	vector<float4> JuA(contacts);
	vector<float4> JvA(contacts);
	vector<float4> JwA(contacts);

	vector<float4> JxB(contacts);
	vector<float4> JyB(contacts);
	vector<float4> JzB(contacts);

	vector<float4> JuB(contacts);
	vector<float4> JvB(contacts);
	vector<float4> JwB(contacts);

	vector<float4> h_g(contacts);

	vector<float4> out_vel_A(contacts * 2);
	vector<float4> out_omg_A(contacts * 2);
	vector<float4> out_vel_B(contacts * 2);
	vector<float4> out_omg_B(contacts * 2);

	// Initialize vectors on host
	int i;
	for (i = 0; i < contacts; i++) {
		JxA[i] = float4(cosf(i) * cosf(i));
		JyA[i] = float4(cosf(i) * cosf(i));
		JzA[i] = float4(cosf(i) * cosf(i));

		h_g[i] = float4(sinf(i) * sinf(i));

		JuA[i] = float4(cosf(i) * cosf(i));
		JvA[i] = float4(cosf(i) * cosf(i));
		JwA[i] = float4(cosf(i) * cosf(i));
		JxB[i] = float4(cosf(i) * cosf(i));
		JyB[i] = float4(cosf(i) * cosf(i));
		JzB[i] = float4(cosf(i) * cosf(i));
		JuB[i] = float4(cosf(i) * cosf(i));
		JvB[i] = float4(cosf(i) * cosf(i));
		JwB[i] = float4(cosf(i) * cosf(i));

	}

	int n_contact = contacts;

	double total_time_omp;
	double total_flops;
	double total_memory;
	double runs = 10;
	for (int i = 0; i < runs; i++) {

		double start = omp_get_wtime();

		Function_1(contacts, h_g.data(), JxA.data(), JuA.data(), JuB.data(), JvA.data(), JvB.data(), JwA.data(), JwB.data(), out_vel_A.data(), out_omg_A.data(), out_vel_B.data(),
				out_omg_B.data());

		double end = omp_get_wtime();

		total_time_omp = (end - start) * 1000;
		total_flops = 12 * contacts / ((end - start)) / 1e9;
		total_memory = (12 * 4 * 4) * contacts / ((end - start)) / 1024.0 / 1024.0 / 1024.0;
		printf("\nExecution time in milliseconds =  %0.3f ms | %0.3f Gflop | %0.3f GB/s \n", total_time_omp, total_flops, total_memory);
	}

	//release host memory

	return 0;
}
