/*
 * test_simd_common_permute.cpp
 *
 *  Created on: May 1, 2020
 *      Author: bjoo
 */



#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "simd.hpp"

TEST(TestSIMDPermute, TestAVX512FloatPermute)
{
	using simd_t = simd::simd<float, simd::simd_abi::avx512>;
	using control_t = simd::simd<int, simd::simd_abi::avx512>;

	using storage_t = typename simd_t::storage_type;

	constexpr int veclen = simd_t::size();
	ASSERT_EQ( veclen,16);

	Kokkos::View< storage_t[1] > simd_vec_in("v_in");
	Kokkos::View< storage_t[1] > simd_vec_out("v_out");
	Kokkos::View< float* >       vec_in_scalar((float *)simd_vec_in.data(),veclen);
	Kokkos::View< float *>       vec_out_scalar((float *)simd_vec_out.data(),veclen);

	for(int i=0; i < veclen; ++i) {
		vec_in_scalar(i) = i;
	}
 	int identity_mask[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	auto identity = simd::simd_utils<simd_t>::make_permute(identity_mask);


	simd_vec_out(0) = simd::permute( control_t(identity), simd_t(simd_vec_in(0)));

	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(i) );
	}
	std::cout << "\n";

	int reverse_mask[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
	auto reverse = simd::simd_utils<simd_t>::make_permute(reverse_mask);
	simd_vec_out(0) = simd::permute( control_t(reverse), simd_t(simd_vec_in(0)) );
	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(veclen-1-i) );
	}
	std::cout << "\n";
}

TEST(TestSIMDPermute, TestAVX512DoublePermute)
{
	using simd_t = simd::simd<double, simd::simd_abi::avx512>;
	using control_t = simd::simd<int, simd::simd_abi::avx512>;

	using storage_t = typename simd_t::storage_type;
	using control_storage_t = typename control_t::storage_type;

	constexpr int veclen = simd_t::size();
	ASSERT_EQ( veclen, 8);


	Kokkos::View< storage_t[1] > simd_vec_in("v_in");
	Kokkos::View< storage_t[1] > simd_vec_out("v_out");
	Kokkos::View< double* >       vec_in_scalar((double *)simd_vec_in.data(),veclen);
	Kokkos::View< double *>       vec_out_scalar((double *)simd_vec_out.data(),veclen);

	for(int i=0; i < veclen; ++i) {
		vec_in_scalar(i) = i;
	}

	int identity_mask[8] = { 0,1,2,3,4,5,6,7};
	auto identity=simd::simd_utils<simd_t>::make_permute(identity_mask);

	simd_vec_out(0) = simd::permute( control_t(identity), simd_t(simd_vec_in(0)));

	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		// ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(i) );
	}
	std::cout << "\n";

	int reverse_mask[8] = {7,6,5,4,3,2,1,0};
	auto reverse=simd::simd_utils<simd_t>::make_permute(reverse_mask);
	simd_vec_out(0) = simd::permute( control_t(reverse), simd_t(simd_vec_in(0)) );
	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(veclen-1-i) );
	}
	std::cout << "\n";
}
