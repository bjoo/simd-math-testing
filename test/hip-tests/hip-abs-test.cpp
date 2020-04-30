#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"

using Kokkos::abs;

#include "simd.hpp"


void absTest()
{
  using simd_t = simd::simd<double,simd::simd_abi::hip_wavefront<64>>;
  using simd_storage_t = simd_t::storage_type;

  int V = simd_t::size();

  Kokkos::View< simd_storage_t[1] > vec_in("vec_in");
  Kokkos::View< simd_storage_t[1] > vec_out("vec_out");

  Kokkos::View<double*> scalar_in((double *)vec_in.data(), V );
  Kokkos::View<double*> scalar_out((double *)vec_out.data(),V);
  
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
    ASSERT_FLOAT_EQ( check_values(i), +1.0);
  }

}

TEST(HIPTests, TestAbs)
{
  absTest();
}
