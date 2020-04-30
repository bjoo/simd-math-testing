# simd-math-testing

This repo collects together some tests of the simd-math module to aid further development and testing.
Configure with

```
cmake -DSIMD_DIR=<path-to-simd-math-headers> \
         -DKokkos_DIR=<path-to-kokkos-install-dir>/lib64/cmake/Kokkos \
         -DCMAKE_CXX_COMPILER=<compiler> \
         -DBUILD_GMOCK=OFF \
         <path-to-simd-math-testing-source-dir>  
```
         
