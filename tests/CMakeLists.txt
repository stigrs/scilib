cmake_minimum_required(VERSION 3.13)
project(Scilib_tests CXX)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

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

if(NOT TARGET std::linalg)
    find_package(linalg REQUIRED)
endif()

if(NOT TARGET scilib::scilib)
    find_package(scilib REQUIRED)
endif()

if(NOT TARGET Microsoft.GSL::GSL)
    find_package(Microsoft.GSL CONFIG)
endif()

if(NOT TARGET range-v3)
    find_package(range-v3 CONFIG)
endif()

if(Scilib_CODE_COVERAGE)
    set(GCOV_LIBRARIES gcov)
endif()

function(add_scilib_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries (
        ${name} 
        PRIVATE
        std::linalg
        std::mdspan
        scilib::scilib
        Microsoft.GSL::GSL
        range-v3
        gtest_main
	    ${BLAS_LIBRARIES}
        ${GCOV_LIBRARIES}
    ) 
    target_compile_options(
        ${name}
        PRIVATE
        "$<$<CONFIG:Debug>:${Scilib_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${Scilib_CXX_FLAGS_RELEASE}>"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

set(PROGRAMS 
    test_vector
    test_matrix
    test_mdspan_iterator
    #test_array3d
    #test_array4d
    # test_integrate
    # test_linalg_aux
    # test_linalg_blas1
    # test_linalg_blas2
    # test_linalg_blas3
    # test_linalg_eigenvalue
    # test_linalg_element_wise_math
    # test_linalg_expm
    # test_linalg_solve
    # test_linalg_lstsq
    # test_linalg_matrix_decomp
    # test_linalg_matrix_det
    # test_linalg_matrix_inv
    # test_linalg_matrix_norm
    # test_linalg_matrix_power
    # test_linalg_trace
    # test_linalg_transposed
    # test_stats
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
