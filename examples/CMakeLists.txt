cmake_minimum_required(VERSION 3.13)
project(Scilib_examples CXX)

include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET std::mdspan)
    find_package(mdspan REQUIRED)
endif()

if(NOT TARGET scilib::scilib)
    find_package(scilib REQUIRED)
endif()

set(PROGRAMS 
    matrix_diag
    mdarray_properties
    solve_ivp
    submatrix_views 
    transposed
)

foreach(program ${PROGRAMS})
    add_executable(${program} ${program}.cpp)
	target_link_libraries (
	    ${program} 
        PRIVATE
        scilib::scilib
        std::mdspan
	    ${BLAS_LIBRARIES}
        OpenMP::OpenMP_CXX
	) 
    target_compile_options(
        ${program}
        PRIVATE
        "$<$<CONFIG:Debug>:${Scilib_CXX_FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${Scilib_CXX_FLAGS_RELEASE}>"
    )
endforeach()
