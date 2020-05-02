cmake_minimum_required(VERSION 3.13)
project(QDPXXKokkosPlay VERSION 0.1
	DESCRIPTION "SIMD-Math Tests"
        LANGUAGES CXX)


include(CTest)

option(ENABLE_AVX_TESTS "Enable AVX Specific tests" FALSE)
option(ENABLE_HIP_TESTS "Enable HIP Specific tests" FALSE)
option(ENABLE_AVX512_TESTS "Enable AVX512 Specific tests" FALSE)
option(ENABLE_CUDA_TESTS "Enable CUDA Specific tests" FALSE)

set(SIMD_DIR "" CACHE STRING "")
set(Kokkos_DIR "" CACHE STRING "")

if( !${SIMD_DIR} )
  message(FATAL, "Must Set SIMD_DIR to path of simd-math")
endif()

if( !${Kokkos_DIR} )
  message(FATAL, "Must Set Kokkos_DIR to path containing KokkosConfig.cmake")
endif()

find_package(Kokkos)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(test)
