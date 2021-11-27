cmake_minimum_required(VERSION 3.14)
project(Scilib_examples CXX)

include_directories(${Scilib_SOURCE_DIR}/include)
include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET std::mdspan)
  find_package(mdspan REQUIRED)
endif()

set(
    PROGRAMS 
    matrix_diag
    submatrix_views 
    transposed
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
        sci
	    ${BLAS_LIBRARIES}
	    ${LAPACK_LIBRARIES}
        std::mdspan
	) 
endforeach()