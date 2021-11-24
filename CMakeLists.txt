cmake_minimum_required(VERSION 3.14)
project(Scilib CXX)

# Set installation directory.
if (WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX $ENV{USERPROFILE} CACHE PATH "USERPROFILE" FORCE)
    endif()
else()
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX $ENV{HOME} CACHE PATH "HOME" FORCE)
    endif()
endif()

# Enforce C++20 standard.
set(CMAKE_CXX_STANDARD 20)

# Export compile commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    option(BUILD_SHARED_LIBS "Compile shared libraries." OFF)
else()
    option(BUILD_SHARED_LIBS "Compile shared libraries." ON)
endif()

option(BUILD_TESTS "Build tests." ON)
option(BUILD_BENCH "Build benchmarks." ON)
option(BUILD_EXAMPLES "Build examples." ON)

# Set default MSVC compiler options to avoid D9025 error.
if(MSVC)
    set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS")
endif()

# Set release compiler options.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-maybe-uninitialized")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/DNDEBUG /W4 /WX /wd4100 /GR /EHsc /MT /O2")
endif()

# Set debugging compiler options.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-maybe-uninitialized")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/EHsc /W4 /WX /wd4100 /MT /GR /Zi /Od")
endif()

# BLAS and LAPACKE are required.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(DEFINED ENV{MKLROOT})
	find_package(MKL)
	if(MKL_FOUND)
		add_definitions(${MKL_DEFINITIONS})
		set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
		set(LAPACKE_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
        set(BLAS_LIBRARY_DIRS ${MKL_LIBRARY_DIRS})
		set(BLAS_LIBRARIES ${MKL_LIBRARIES})
	endif()
else()
    find_package(OpenBLAS)
    add_definitions(${BLAS_DEFINITIONS})
endif()

install(
    DIRECTORY include/scilib
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

#add_subdirectory(lib)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCH)
    # Armadillo library is required.
    if(WIN32)
		set(ARMA_DIR C:/local/armadillo-10.7.3)
        find_file(ARMADILLO_LIBRARIES armadillo.lib HINTS ${ARMA_DIR}/lib)
        find_path(ARMADILLO_INCLUDE_DIRS armadillo HINTS ${ARMA_DIR}/include)
        add_definitions(-DARMA_NO_DEBUG)
    else()
        find_package(Armadillo 7.900.1 REQUIRED)
        add_definitions(-DARMA_NO_DEBUG)
    endif()
	add_subdirectory(bench)
endif()

if(BUILD_EXAMPLES) 
    add_subdirectory(examples)
endif()
