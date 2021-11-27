cmake_minimum_required(VERSION 3.14)
project(libsci CXX)

include_directories(${Scilib_SOURCE_DIR}/include)
include_directories(${BLAS_INCLUDE_DIRS})
include_directories(${LAPACKE_INCLUDE_DIRS})
link_directories(${Scilib_BINARY_DIR})
link_directories(${BLAS_LIBRARY_DIRS})

if(NOT TARGET std::mdspan)
  find_package(mdspan REQUIRED)
endif()

set(
    SRC_FILES
    eigenvalue.cpp
)
add_library(
    sci 
    ${SRC_FILES}
)
target_link_libraries(
    sci 
    ${BLAS_LIBRARIES}
    std::mdspan
) 

install(
    TARGETS sci
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
