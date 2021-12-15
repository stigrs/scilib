cmake_minimum_required(VERSION 3.14)
project(Scilib_bench CXX)

include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

if(NOT TARGET Scilib::Scilib)
  find_package(scilib REQUIRED)
endif()

set(PROGRAMS 
    bench_axpy 
    bench_dot 
    bench_eigs
    bench_matrix_product
    bench_matrix_vector_product
    bench_transpose
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
        Scilib::Scilib
        std::mdspan
        Eigen3::Eigen
	    ${BLAS_LIBRARIES}
	) 
    target_include_directories(scilib PUBLIC
        $<BUILD_INTERFACE:${mdspan_SOURCE_DIR}>
    )
endforeach()
