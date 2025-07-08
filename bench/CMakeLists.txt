cmake_minimum_required(VERSION 3.13)
project(Scilib_bench CXX)

include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET mdspan::mdspan)
    find_package(mdspan REQUIRED)
endif()

if(NOT TARGET std::linalg)
    find_package(linalg REQUIRED)
endif()

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
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

set(PROGRAMS 
    bench_axpy 
    bench_dot 
    bench_eigh
    bench_finite_difference
    bench_matrix_product
    bench_matrix_vector_product
    bench_mkl_dgemm
    bench_transpose
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
        PRIVATE
        scilib::scilib
        mdspan::mdspan
        std::linalg
        range-v3
        Microsoft.GSL::GSL
        Eigen3::Eigen
	    ${BLAS_LIBRARIES}
	) 
    target_compile_options(
        ${program}
        PRIVATE
        "$<$<CONFIG:Debug>:${Scilib_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${Scilib_CXX_FLAGS_RELEASE}>"
    )
endforeach()
