cmake_minimum_required(VERSION 3.17)
project(Scilib_tests CXX)

include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

include(FetchContent)
FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(NOT TARGET std::mdspan)
    find_package(mdspan REQUIRED)
endif()

if(NOT TARGET Scilib::Scilib)
    find_package(scilib REQUIRED)
endif()

function(add_scilib_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries (
        ${name} 
        Scilib::Scilib
        gtest_main
        std::mdspan
	    ${BLAS_LIBRARIES}
    ) 
    add_test(NAME ${name} COMMAND ${name})
endfunction()

set(PROGRAMS 
    test_vector
    test_matrix
    test_array3d
    test_array4d
    test_integrate
    test_linalg_blas1
    test_linalg_blas2
    test_linalg_blas3
    test_linalg_eigenvalue
    test_linalg_element_wise_math
    test_linalg_expm
    test_linalg_linsolve
    test_linalg_lstsq
    test_linalg_matrix_decomp
    test_linalg_matrix_det
    test_linalg_matrix_inv
    test_linalg_matrix_power
    test_linalg_transposed
    test_stats
)

foreach(program ${PROGRAMS})
    add_scilib_test(${program})
	if(WIN32)
	    if(NOT MKL_FOUND)
          set_tests_properties(${program} PROPERTIES ENVIRONMENT "PATH=${BLAS_BINARY_DIRS};$ENV{PATH}")
	    endif()
  else()
	    if(NOT MKL_FOUND)
          set_tests_properties(${program} PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${BLAS_BINARY_DIRS}")
	    endif()
	endif()
endforeach()
