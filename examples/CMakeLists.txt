cmake_minimum_required(VERSION 3.14)
project(Scilib_examples CXX)

include_directories(${Scilib_SOURCE_DIR}/include)
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
    submatrix_views 
    matrix_diag
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
	    ${BLAS_LIBRARIES}
	    ${LAPACK_LIBRARIES}
        std::mdspan
	) 
endforeach()