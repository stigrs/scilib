cmake_minimum_required(VERSION 3.13)
project(Scilib_bench CXX)

include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET std::mdspan)
    find_package(mdspan REQUIRED)
endif()

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

if(NOT TARGET scilib::scilib)
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
        PRIVATE
        scilib::scilib
        std::mdspan
        Eigen3::Eigen
	    ${BLAS_LIBRARIES}
	) 
    target_compile_options(
        ${program}
        PRIVATE
        "$<$<CONFIG:Debug>:${Stdutils_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${Stdutils_CXX_FLAGS_RELEASE}>"
    )
endforeach()
