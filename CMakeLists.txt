cmake_minimum_required(VERSION 3.14)
project(Scilib 
    VERSION 0.1
    LANGUAGES CXX
)

################################################################################

include(GNUInstallDirs)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################

option(BUILD_TESTS "Build tests." ON)
option(BUILD_BENCH "Build benchmarks." OFF)
option(BUILD_EXAMPLES "Build examples." OFF)

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

# Option to override which C++ standard to use
set(Scilib_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE Scilib_CXX_STANDARD PROPERTY STRINGS DETECT 17 20)

option(Scilib_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." ON)

# Decide on the standard to use
if(Scilib_CXX_STANDARD STREQUAL "17")
    if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++17 standard")
        set(CMAKE_CXX_STANDARD 17)
    else()
        message(FATAL_ERROR "Requested Scilib_CXX_STANDARD \"17\" not supported by provided C++ compiler")
    endif()
elseif(Scilib_CXX_STANDARD STREQUAL "20")
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++20 standard")
        set(CMAKE_CXX_STANDARD 20)
    else()
        message(FATAL_ERROR "Requested Scilib_CXX_STANDARD \"20\" not supported by provided C++ compiler")
    endif()
else()
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "Detected support for C++20 standard")
    elseif("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "Detected support for C++17 standard")
    else()
        message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++14 or newer.")
    endif()
endif()

if(Scilib_ENABLE_CONCEPTS)
    if(CMAKE_CXX_STANDARD STREQUAL "20")
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-fconcepts" COMPILER_SUPPORTS_FCONCEPTS)
        if(COMPILER_SUPPORTS_FCONCEPTS)
            message(STATUS "-- Using \"-fconcepts\" to enable concepts support")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
        else()
            CHECK_CXX_COMPILER_FLAG("-fconcepts-ts" COMPILER_SUPPORTS_FCONCEPTS_TS)
            if(COMPILER_SUPPORTS_FCONCEPTS)
                message(STATUS "-- Using \"-fconcepts-ts\" to enable concepts support")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-ts")
            endif()
        endif()
        # Otherwise, it's possible that the compiler supports concepts without flags,
        # but if it doesn't, they just won't be used, which is fine
    endif()
endif()

# Export compile commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    option(BUILD_SHARED_LIBS "Compile shared libraries." OFF)
else()
    option(BUILD_SHARED_LIBS "Compile shared libraries." ON)
endif()

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

################################################################################

# mdspan is required.
find_package(mdspan QUIET)
if (NOT mdspan_FOUND)
    message(STATUS "No installed mdspan found, fetching from Github")
    include(FetchContent)
    FetchContent_Declare(
        mdspan
        URL https://github.com/kokkos/mdspan/blob/single-header/mdspan.hpp
        DOWNLOAD_NO_EXTRACT TRUE
    )
    FetchContent_GetProperties(mdspan)
    if(mdspan_POPULATED)
        return()
    endif()
    FetchContent_Populate(mdspan)
endif()

# BLAS and LAPACKE are required.
if(DEFINED ENV{MKLROOT})
	find_package(MKL)
	if(MKL_FOUND)
	    message(STATUS "Intel MKL found")
		add_definitions(${MKL_DEFINITIONS})
	endif()
else()
    find_package(OpenBLAS)
	if(BLAS_FOUND)
        MESSAGE(STATUS "OpenBLAS found")
		add_definitions(${BLAS_DEFINITIONS})
	endif()
endif()

################################################################################

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/det.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dormand_prince.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eigenvalue.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/expm.cpp    
)

add_library(scilib ${SRC_FILES})
add_library(Scilib::Scilib ALIAS scilib)
target_link_libraries(
    scilib 
    std::mdspan
    ${BLAS_LIBRARIES}
)

target_include_directories(scilib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${mdspan_SOURCE_DIR}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${LAPACKE_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS scilib
    EXPORT ScilibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}    
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/scilib DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ScilibTargets
    FILE ScilibTargets.cmake
    NAMESPACE Scilib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ScilibConfigVersion.cmake"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ScilibConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/ScilibConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)
install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/ScilibConfig.cmake" 
    "${CMAKE_CURRENT_BINARY_DIR}/ScilibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scilib
)

################################################################################

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCH)
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

if(BUILD_EXAMPLES) 
    add_subdirectory(examples)
endif()
