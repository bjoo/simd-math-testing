#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"

#include "simd.hpp"
//
// ABS
template<typename T>
void absTest()
{
  using simd_t = simd::simd<T,simd::simd_abi::hip_wavefront<64>>;
  using simd_storage_t = typename simd_t::storage_type;

  int V = simd_t::size();

  Kokkos::View< simd_storage_t[1] > vec_in("vec_in");
  Kokkos::View< simd_storage_t[1] > vec_out("vec_out");

  Kokkos::View<T*> scalar_in((T *)vec_in.data(), V );
  Kokkos::View<T*> scalar_out((T *)vec_out.data(),V);
  
  // Set it up -- this should fill vec in and vec out 
  Kokkos::deep_copy(scalar_in,-1.0);
  Kokkos::deep_copy(scalar_out, 0.0);
 
  Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(1,1,simd_t::size()),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
			 vec_out(0) = simd::abs(simd_t(vec_in(0)));
		       });
  Kokkos::fence();

  auto check_values = Kokkos::create_mirror_view(scalar_out);
  Kokkos::deep_copy(check_values, scalar_out);
  for(int i=0; i < simd_t::size(); ++i) {
    if ( sizeof(T) == sizeof(float) ) { 
      ASSERT_FLOAT_EQ( check_values(i), +1.0f);
    }
    else {
      ASSERT_DOUBLE_EQ(check_values(i), +1.0);
    }
  }
}

TEST(HIPTests, TestAbs)
{
  absTest<float>();
  absTest<double>();
}

//
// SQRT
template<typename T>
void sqrtTest()
{
  using simd_t = simd::simd<T,simd::simd_abi::hip_wavefront<64>>;
  using simd_storage_t = typename simd_t::storage_type;

  int V = simd_t::size();

  Kokkos::View< simd_storage_t[1] > vec_in("vec_in");
  Kokkos::View< simd_storage_t[1] > vec_out("vec_out");

  Kokkos::View<T*> scalar_in((T *)vec_in.data(), V );
  Kokkos::View<T*> scalar_out((T *)vec_out.data(),V);
  
  // Set it up -- this should fill vec in and vec out 
  Kokkos::deep_copy(scalar_in,4.0);
  Kokkos::deep_copy(scalar_out, 0.0);
 
  Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(1,1,simd_t::size()),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
			 vec_out(0) = simd::sqrt(simd_t(vec_in(0)));
		       });
  Kokkos::fence();

  auto check_values = Kokkos::create_mirror_view(scalar_out);
  Kokkos::deep_copy(check_values, scalar_out);
  for(int i=0; i < simd_t::size(); ++i) {
    if ( sizeof(T) == sizeof(float) ) { 
      ASSERT_FLOAT_EQ( check_values(i), +2.0f);
    }
    else {
      ASSERT_DOUBLE_EQ(check_values(i), +2.0);
    }
  }
}

TEST(HIPTests, TestSqrt)
{
  sqrtTest<float>();
  // sqrtTest<double>();
}

//
// SQRT
template<typename T>
void expTest()
{
  using simd_t = simd::simd<T,simd::simd_abi::hip_wavefront<64>>;
  using simd_storage_t = typename simd_t::storage_type;

  int V = simd_t::size();

  Kokkos::View< simd_storage_t[1] > vec_in("vec_in");
  Kokkos::View< simd_storage_t[1] > vec_out("vec_out");

  Kokkos::View<T*> scalar_in((T *)vec_in.data(), V );
  Kokkos::View<T*> scalar_out((T *)vec_out.data(),V);
  
  // Set it up -- this should fill vec in and vec out 
  Kokkos::deep_copy(scalar_in,4.0);
  Kokkos::deep_copy(scalar_out, 0.0);
 
  Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(1,1,simd_t::size()),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
			 vec_out(0) = simd::exp(simd_t(vec_in(0)));
		       });
  Kokkos::fence();

  auto check_values = Kokkos::create_mirror_view(scalar_out);
  Kokkos::deep_copy(check_values, scalar_out);
  for(int i=0; i < simd_t::size(); ++i) {
    if ( sizeof(T) == sizeof(float) ) { 
      ASSERT_FLOAT_EQ( check_values(i), std::exp(4.0f));
    }
    else {
      ASSERT_DOUBLE_EQ(check_values(i), std::exp(4.0));
    }
  }
}

TEST(HIPTests, TestExp)
{
  expTest<float>();
  // expTest<double>();
}
//
// SQRT
template<typename T>
void cbrtTest()
{
  using simd_t = simd::simd<T,simd::simd_abi::hip_wavefront<64>>;
  using simd_storage_t = typename simd_t::storage_type;

  int V = simd_t::size();

  Kokkos::View< simd_storage_t[1] > vec_in("vec_in");
  Kokkos::View< simd_storage_t[1] > vec_out("vec_out");

  Kokkos::View<T*> scalar_in((T *)vec_in.data(), V );
  Kokkos::View<T*> scalar_out((T *)vec_out.data(),V);
  
  // Set it up -- this should fill vec in and vec out 
  Kokkos::deep_copy(scalar_in,4.0);
  Kokkos::deep_copy(scalar_out, 0.0);
 
  Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(1,1,simd_t::size()),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
			 vec_out(0) = simd::cbrt(simd_t(vec_in(0)));
		       });
  Kokkos::fence();

  auto check_values = Kokkos::create_mirror_view(scalar_out);
  Kokkos::deep_copy(check_values, scalar_out);
  for(int i=0; i < simd_t::size(); ++i) {
    if ( sizeof(T) == sizeof(float) ) { 
      ASSERT_FLOAT_EQ( check_values(i), std::cbrt(4.0f));
    }
    else {
      ASSERT_DOUBLE_EQ(check_values(i), std::cbrt(4.0));
    }
  }
}

TEST(HIPTests, TestCBRT)
{
  cbrtTest<float>();
  //cbrtTest<double>();
}
