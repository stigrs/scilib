cmake_minimum_required(VERSION 3.13)
project(Scilib 
    VERSION 0.3
    LANGUAGES CXX
)

################################################################################

include(GNUInstallDirs)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################

set(Scilib_STANDALONE_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(Scilib_STANDALONE_PROJECT ON)
endif()

option(Scilib_BUILD_TESTS "Build tests." ${Scilib_STANDALONE_PROJECT})
option(Scilib_BUILD_BENCH "Build benchmarks." ${Scilib_STANDALONE_PROJECT})
option(Scilib_BUILD_EXAMPLES "Build examples." ${Scilib_STANDALONE_PROJECT})
option(Scilib_CODE_COVERAGE "Enable code coverage." OFF)

################################################################################

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

################################################################################

if(WIN32)
    option(BUILD_SHARED_LIBS "Compile shared libraries." OFF)
else()
    option(BUILD_SHARED_LIBS "Compile shared libraries." ON)
endif()

# Check if C++23 standard is available.
if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
else()
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++20 standard")
        set(CMAKE_CXX_STANDARD 20)
    else()
        message(FATAL_ERROR "Requested Scilib_CXX_STANDARD not supported by provided C++ compiler")
    endif()
endif()

option(Scilib_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." ON)

if(Scilib_ENABLE_CONCEPTS)
    if(CMAKE_CXX_STANDARD GREATER_EQUAL 20)
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-fconcepts" COMPILER_SUPPORTS_FCONCEPTS)
        if(COMPILER_SUPPORTS_FCONCEPTS)
            message(STATUS "-- Using \"-fconcepts\" to enable concepts support")
            set(Scilib_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fconcepts)
        else()
            CHECK_CXX_COMPILER_FLAG("-fconcepts-ts" COMPILER_SUPPORTS_FCONCEPTS_TS)
            if(COMPILER_SUPPORTS_FCONCEPTS)
                message(STATUS "-- Using \"-fconcepts-ts\" to enable concepts support")
                set(Scilib_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fconcepts-ts)
            endif()
        endif()
    endif()
endif()

# Set compiler options.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(Scilib_CXX_FLAGS ${Scilib_CXX_FLAGS} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-error=unused-parameter -Wno-maybe-uninitialized)
    set(Scilib_CXX_FLAGS_DEBUG ${Scilib_CXX_FLAGS} -g)
    set(Scilib_CXX_FLAGS_RELEASE ${Scilib_CXX_FLAGS} -O3)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(Scilib_CMAKE_CXX_FLAGS ${Scilib_CXX_FLAGS} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow)
    set(Scilib_CXX_FLAGS_DEBUG ${Scilib_CXX_FLAGS} -g)
    set(Scilib_CXX_FLAGS_RELEASE ${Scilib_CXX_FLAGS} -O3)
endif()
if(MSVC)
    if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        set(Scilib_CXX_FLAGS ${Scilib_CXX_FLAGS} /W4 /Qstd:c++20)
        set(Scilib_CXX_FLAGS_DEBUG ${Scilib_CXX_FLAGS} /Zi /Od /Qopenmp)
        set(Scilib_CXX_FLAGS_RELEASE ${Scilib_CXX_FLAGS} /O3 /Qopenmp)
    else()
        set(Scilib_CXX_FLAGS ${Scilib_CXX_FLAGS} /W4 /WX) 
        set(Scilib_CXX_FLAGS_DEBUG ${Scilib_CXX_FLAGS} /Zi /Od)
        set(Scilib_CXX_FLAGS_RELEASE ${Scilib_CXX_FLAGS} /O2)
    endif()
endif()
if (Scilib_CODE_COVERAGE)
    set(Scilib_CXX_FLAGS ${Scilib_CXX_FLAGS} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-error=unused-parameter -Wno-maybe-uninitialized)
    set(Scilib_CXX_FLAGS_DEBUG ${Scilib_CXX_FLAGS} -O0 -g -fprofile-arcs -ftest-coverage)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################################################################

# mdspan is required.
find_package(mdspan QUIET)
if (NOT mdspan_FOUND)
    message(STATUS "No installed mdspan found, fetching from Github")
    include(FetchContent)
    FetchContent_Declare(
        mdspan
        GIT_REPOSITORY  https://github.com/kokkos/mdspan.git
        GIT_TAG         stable
    )
    FetchContent_MakeAvailable(mdspan)
endif()

if(NOT TARGET std::mdspan)
    find_package(mdspan REQUIRED)
endif()

# stdBLAS is required.
message(STATUS "Fetch stdBLAS from Github")
include(FetchContent)
FetchContent_Declare(
    linalg
    GIT_REPOSITORY  https://github.com/stigrs/stdBLAS.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(linalg)

if(NOT TARGET std::linalg)
    find_package(linalg REQUIRED)
endif()

# GSL is required.
message(STATUS "Fetch GSL from Github")
include(FetchContent)
FetchContent_Declare(
    GSL
    GIT_REPOSITORY  https://github.com/microsoft/GSL.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(GSL)

if(NOT TARGET Microsoft.GSL::GSL)
    find_package(Microsoft.GSL CONFIG)
endif()

# BLAS and LAPACKE are required.
if(DEFINED ENV{MKLROOT})
	find_package(MKL)
	if(MKL_FOUND)
	    message(STATUS "Intel MKL found")
	    add_definitions(${MKL_DEFINITIONS})
	endif()
    option(USE_MKL_ALLOCATOR OFF)
    option(MKL_MEM_ALIGNMENT_VALUE "")
    if(USE_MKL_ALLOCATOR)
        if(MSVC)
            add_definitions(/DUSE_MKL_ALLOCATOR /DMKL_ALIGN)
            if(MKL_MEM_ALIGNMENT_VALUE)
                add_definitions(/DMKL_MEM_ALIGNMENT=${MKL_MEM_ALIGNMENT_VALUE})
            endif()
        else()
            add_definitions(-DUSE_MKL_ALLOCATOR -DMKL_ALIGN)
            if(MKL_MEM_ALIGNMENT_VALUE)
                add_definitions(-DMKL_MEM_ALIGNMENT=${MKL_MEM_ALIGNMENT_VALUE})
            endif()
        endif()
    endif()
else()
    find_package(OpenBLAS)
	if(BLAS_FOUND)
        MESSAGE(STATUS "OpenBLAS found")
	    add_definitions(${BLAS_DEFINITIONS})
	endif()
endif()

################################################################################

add_library(scilib INTERFACE)
add_library(scilib::scilib ALIAS scilib)

target_link_libraries(scilib INTERFACE std::mdspan std::linalg Microsoft.GSL::GSL)

target_include_directories(scilib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${LAPACKE_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS scilib mdspan linalg GSL
    EXPORT scilibTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}    
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/scilib DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT scilibTargets
    FILE scilibTargets.cmake
    NAMESPACE scilib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/scilibConfigVersion.cmake"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/scilibConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/scilibConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)
install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/scilibConfig.cmake" 
    "${CMAKE_CURRENT_BINARY_DIR}/scilibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)

################################################################################

if(Scilib_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(Scilib_BUILD_BENCH)
    find_package(Eigen3 3.4 QUIET)
    if (NOT TARGET Eigen3::Eigen)
        message(STATUS "No installed Eigen found, fetching from GitLab")
        include(FetchContent)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        master
        )
        FetchContent_MakeAvailable(eigen)
    endif()
    add_subdirectory(bench)
endif()

if(Scilib_BUILD_EXAMPLES) 
    add_subdirectory(examples)
endif()
