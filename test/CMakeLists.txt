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

# Here TESTNAME is test1
# and ARGN is test1.cpp
package_add_test(tut_ex9 ./exercise_9_solution.cpp)
package_add_test(tut_ex10 ./exercise_10_solution.cpp)
package_add_test(hip_tests ./hip-tests/hip-abs-test.cpp)
