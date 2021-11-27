cmake_minimum_required(VERSION 3.14)
project(Scilib 
    VERSION 0.0.1
    LANGUAGES CXX
)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################

option(BUILD_TESTS "Build tests." ON)
option(BUILD_BENCH "Build benchmarks." ON)
option(BUILD_EXAMPLES "Build examples." ON)

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
set(MDSPAN_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE MDSPAN_CXX_STANDARD PROPERTY STRINGS DETECT 14 17 20)

option(MDSPAN_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." ON)

# Decide on the standard to use
if(MDSPAN_CXX_STANDARD STREQUAL "17")
    if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++17 standard")
        set(CMAKE_CXX_STANDARD 17)
    else()
        message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"17\" not supported by provided C++ compiler")
    endif()
elseif(MDSPAN_CXX_STANDARD STREQUAL "14")
    if("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++14 standard")
        set(CMAKE_CXX_STANDARD 14)
    else()
        message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"14\" not supported by provided C++ compiler")
    endif()
elseif(MDSPAN_CXX_STANDARD STREQUAL "20")
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(STATUS "Using C++20 standard")
        set(CMAKE_CXX_STANDARD 20)
    else()
        message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"20\" not supported by provided C++ compiler")
    endif()
else()
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "Detected support for C++20 standard")
    elseif("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "Detected support for C++17 standard")
    elseif("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "Detected support for C++14 standard")
    else()
        message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++14 or newer.")
    endif()
endif()

if(MDSPAN_ENABLE_CONCEPTS)
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
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG        stable
    )
    FetchContent_GetProperties(mdspan)
    if(NOT mdspan_POPULATED)
        FetchContent_Populate(mdspan)
        add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
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

#add_subdirectory(lib)

add_library(scilib INTERFACE)
target_link_libraries(scilib INTERFACE std::mdspan)

target_include_directories(scilib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

################################################################################

install(TARGETS scilib EXPORT ScilibTargets
  INCLUDES DESTINATION include
)

install(EXPORT ScilibTargets
    FILE ScilibTargets.cmake
    NAMESPACE Scilib::
    DESTINATION lib/cmake/scilib
)

export(TARGETS scilib
    NAMESPACE Scilib::
    FILE ScilibTargets.cmake
)

install(DIRECTORY include/scilib DESTINATION include)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/ScilibConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ScilibConfig.cmake
  INSTALL_DESTINATION lib/cmake/scilib
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ScilibConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ScilibConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/ScilibConfigVersion.cmake
    DESTINATION lib/cmake/scilib
)

################################################################################

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
        add_definitions(-DARMA_NO_DEBUG -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK)
    endif()
	add_subdirectory(bench)
endif()

if(BUILD_EXAMPLES) 
    add_subdirectory(examples)
endif()
