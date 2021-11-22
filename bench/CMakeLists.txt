cmake_minimum_required(VERSION 3.14)
project(Scilib_bench CXX)

if(WIN32)
    include_directories(${ARMADILLO_INCLUDE_DIRS})
endif()
include_directories(${Scilib_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
if(MKL_FOUND)
    link_directories(${MKL_LIBRARY_DIRS})
endif()

if(NOT TARGET std::mdspan)
  find_package(mdspan REQUIRED)
endif()

set(
    PROGRAMS 
    bench_axpy 
    #bench_dot 
    #bench_eigs
    #bench_mm_mul
    #bench_mv_mul
    #bench_norm
    #bench_transpose
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
	    ${BLAS_LIBRARIES}
	    ${LAPACK_LIBRARIES}
	    ${ARMADILLO_LIBRARIES} 
        std::mdspan
	) 
endforeach()
