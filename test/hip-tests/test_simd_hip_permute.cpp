/*
 * test_simd_common_permute.cpp
 *
 *  Created on: May 1, 2020
 *      Author: bjoo
 */



#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "simd.hpp"

template<int veclen, typename T>
void testGPUSIMD()
{
  using simd_t = simd::simd<T, simd::simd_abi::hip_wavefront<veclen>>;
  using control_t = simd::simd<int, simd::simd_abi::hip_wavefront<veclen>>;

  using storage_t = typename simd_t::storage_type;
  using control_storage_t = typename control_t::storage_type;
  
  ASSERT_EQ( veclen, simd_t::size());


  Kokkos::View< storage_t[1] > simd_vec_in("v_in");
  Kokkos::View< storage_t[2] > simd_vec_out("v_out");
  Kokkos::View< T* >       vec_in_scalar((T *)simd_vec_in.data(),veclen);
  Kokkos::View< T**>       vec_out_scalar((T *)simd_vec_out.data(),veclen,2);

  auto h_vec_in_scalar = Kokkos::create_mirror_view(vec_in_scalar);
  
  for(int i=0; i < veclen; ++i) {
    h_vec_in_scalar(i) = i;
  }
  Kokkos::deep_copy(vec_in_scalar,h_vec_in_scalar);
  
  control_storage_t identity;
  for(int i=0; i < veclen; ++i) identity[i]=i;
  control_storage_t reverse;
  for(int i=0; i < veclen; ++i) reverse[i] =(veclen-1)-i;
  
  Kokkos::parallel_for("SIMD", Kokkos::TeamPolicy<>(1,1,simd_t::size()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {      
      simd_vec_out(0) = simd::permute( control_t(identity), simd_t(simd_vec_in(0)));
      simd_vec_out(1) = simd::permute( control_t(reverse), simd_t(simd_vec_in(0)) );
		       });

  Kokkos::fence();
  
  auto h_vec_out_scalar = Kokkos::create_mirror_view(vec_out_scalar);
  Kokkos::deep_copy(h_vec_out_scalar,vec_out_scalar);

  for(int i=0; i < veclen; ++i) {
    std::cout << h_vec_out_scalar(i,0) << " ";
    if ( sizeof(T) == sizeof(float ) ) ASSERT_FLOAT_EQ( h_vec_out_scalar(i,0), h_vec_in_scalar(i) );
    if ( sizeof(T) == sizeof(double ) ) ASSERT_DOUBLE_EQ( h_vec_out_scalar(i,0), h_vec_in_scalar(i) );

  }
  std::cout << "\n";
  
  for(int i=0; i < veclen; ++i) {
    std::cout << h_vec_out_scalar(i,1) << " ";
    if( sizeof(T) == sizeof(float) ) ASSERT_FLOAT_EQ( h_vec_out_scalar(i,1), h_vec_in_scalar(veclen-1-i) );
    if( sizeof(T) == sizeof(double) ) ASSERT_DOUBLE_EQ( h_vec_out_scalar(i,1), h_vec_in_scalar(veclen-1-i) );
  }
  std::cout << "\n";
}

TEST(TestSIMDPermute, TestHIPFloatPermute)
{

#ifdef __HCC__
  testGPUSIMD<64,float>();
#endif

  testGPUSIMD<32,float>();
  testGPUSIMD<16,float>();
  testGPUSIMD<8,float>();
  testGPUSIMD<4,float>();
  testGPUSIMD<2,float>();
}

TEST(TestSIMDPermute, TestHIPDoublePermute)
{
#ifdef __HCC__
  testGPUSIMD<64,double>();
#endif

  testGPUSIMD<32,double>();
  testGPUSIMD<16,double>();
  testGPUSIMD<8,double>();
  testGPUSIMD<4,double>();
  testGPUSIMD<2,double>();
}
