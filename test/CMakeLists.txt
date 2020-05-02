# Add the Googletest directory and mark its targets etc as advanced


add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
set_target_properties(gtest PROPERTIES FOLDER extern)

add_library( testutils test_env.cpp )
target_link_libraries( testutils Kokkos::kokkos gtest )

# This macro takes the testname and atts an executable from the argumnets
# (ARGN is every argument after the listed one) which allows you to pass a list
# of sources after the testname
macro(package_add_test TESTNAME)
    # Make the executable
    add_executable(${TESTNAME} ${ARGN})
    target_include_directories(${TESTNAME} PUBLIC ${SIMD_DIR})

    # Add on the gtest libs
    target_link_libraries(${TESTNAME} testutils)

    # Add the test to CTest
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})

    # I am not 100% sure what tis is for but it is a property
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

# General tests (simd-common)
package_add_test(tut_ex9 ./exercise_9_solution.cpp)

if( ${ENABLE_AVX_TESTS} )
package_add_test(permute ./avx-tests/test_simd_avx_permute.cpp)
endif()


# CUDA Specific tests ( simd_abi::cuda_warp<32> )
if( ${ENABLE_CUDA_TESTS} )
package_add_test(tut_ex10 ./cuda-tests/exercise_10_solution.cpp)
endif()

# HIP Specific Tests ( simd_abi::hip_wavefront<64> )
if( ${ENABLE_HIP_TESTS} ) 
package_add_test(hip_tests ./hip-tests/hip-tests.cpp)
endif()