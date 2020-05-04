/*
 * test_simd_common_permute.cpp
 *
 *  Created on: May 1, 2020
 *      Author: bjoo
 */



#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "simd.hpp"

TEST(TestSIMDPermute, TestAVXFloatPermute)
{
	using simd_t = simd::simd<float, simd::simd_abi::avx>;
	using control_t = simd::simd<int, simd::simd_abi::avx>;

	using storage_t = typename simd_t::storage_type;

	constexpr int veclen = simd_t::size();
	ASSERT_EQ( veclen, 8);


	Kokkos::View< storage_t[1] > simd_vec_in("v_in");
	Kokkos::View< storage_t[1] > simd_vec_out("v_out");
	Kokkos::View< float* >       vec_in_scalar((float *)simd_vec_in.data(),veclen);
	Kokkos::View< float *>       vec_out_scalar((float *)simd_vec_out.data(),veclen);

	for(int i=0; i < 8; ++i) {
		vec_in_scalar(i) = i;
	}
  
	int identity_mask[8] = {0,1,2,3,4,5,6,7}; 
	auto identity = simd::simd_utils<simd_t>::make_permute(identity_mask);

	simd_vec_out(0) = simd::permute( control_t(identity), simd_t(simd_vec_in(0)));

	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(i) );
	}
	std::cout << "\n";

	int reverse_mask[8] = {7,6,5,4,3,2,1,0};
	auto reverse = simd::simd_utils<simd_t>::make_permute(reverse_mask);

	simd_vec_out(0) = simd::permute( control_t(reverse), simd_t(simd_vec_in(0)) );
	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(veclen-1-i) );
	}
	std::cout << "\n";
}

TEST(TestSIMDPermute, TestAVXDoublePermute)
{
	using simd_t = simd::simd<double, simd::simd_abi::avx>;
	using control_t = simd::simd<int, simd::simd_abi::avx>;

	using storage_t = typename simd_t::storage_type;
	using control_storage_t = typename control_t::storage_type;

	constexpr int veclen = simd_t::size();
	ASSERT_EQ( veclen, 4);


	Kokkos::View< storage_t[1] > simd_vec_in("v_in");
	Kokkos::View< storage_t[1] > simd_vec_out("v_out");
	Kokkos::View< double* >       vec_in_scalar((double *)simd_vec_in.data(),veclen);
	Kokkos::View< double *>       vec_out_scalar((double *)simd_vec_out.data(),veclen);

	for(int i=0; i < 4; ++i) {
		vec_in_scalar(i) = i;
	}

	int identity_mask[4] = {0,1,2,3};
	auto identity = simd::simd_utils<simd_t>::make_permute(identity_mask);

	simd_vec_out(0) = simd::permute( control_t(identity), simd_t(simd_vec_in(0)));

	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(i) );
	}
	std::cout << "\n";

	int reverse_mask[4] = {3,2,1,0} ;
	auto reverse = simd::simd_utils<simd_t>::make_permute(reverse_mask);

	simd_vec_out(0) = simd::permute( control_t(reverse), simd_t(simd_vec_in(0)) );
	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(veclen-1-i) );
	}
	std::cout << "\n";
}
#if 0
TEST(TestSIMDPermute, TestAVXDoublePermute)
{
	using simd_t = simd::simd<double, simd::simd_abi::avx>;
	using storage_t = simd_t::storage_type;

	constexpr int veclen = simd_t::size();
	ASSERT_EQ( veclen, 4);

	using control_type_f = simd::simd_permute_control<double,simd::simd_abi::avx>;


	Kokkos::View< storage_t[1] > simd_vec_in("v_in");
	Kokkos::View< storage_t[1] > simd_vec_out("v_out");
	Kokkos::View< double* >       vec_in_scalar((double *)simd_vec_in.data(),veclen);
	Kokkos::View< double *>       vec_out_scalar((double *)simd_vec_out.data(),veclen);

	for(int i=0; i < 8; ++i) {
		vec_in_scalar(i) = i;
	}

	// This is not appropriate in kokkos but we can do it here because
	// OpenMP space is the same as host space (access simd view directly
	// Can permute the lanes directly without a team construct, since it just calls intrinsics
	control_type_f ctrl_f_identity(3,2,1,0);

	simd_vec_out(0) = simd::permute( simd_t(simd_vec_in(0)), ctrl_f_identity);

	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(i) );
	}
	std::cout << "\n";

	control_type_f ctrl_f_reverse(0,1,2,3);
	simd_vec_out(0) = simd::permute( simd_t(simd_vec_in(0)), ctrl_f_reverse);
	for(int i=0; i < veclen; ++i) {
		std::cout << vec_out_scalar(i) << " ";
		ASSERT_FLOAT_EQ( vec_out_scalar(i), vec_in_scalar(veclen-1-i) );
	}
	std::cout << "\n";

}
#endif
