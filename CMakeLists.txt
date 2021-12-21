cmake_minimum_required(VERSION 3.13)
project(Scilib 
    VERSION 0.1
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

# Check if C++20 standard is available.
if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
else()
    message(FATAL_ERROR "Requested Scilib_CXX_STANDARD \"20\" not supported by provided C++ compiler")
endif()

option(Scilib_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." ON)

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
    endif()
endif()

# Set release compiler options.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-maybe-uninitialized -std=c++2a")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -std=c++2a")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/DWIN32 /D_WINDOWS /DNDEBUG /W4 /WX /wd4100 /GR /EHsc /MT /O2 /std:c++latest")
endif()

# Set debugging compiler options.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -Wno-maybe-uninitialized -std=c++2a")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wpedantic -Wall -Wshadow -Wextra -Wno-missing-braces -Werror -Wno-error=strict-overflow -std=c++2a")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/DWIN32 /D_WINDOWS /EHsc /W4 /WX /wd4100 /MT /GR /Zi /Od /std:c++latest")
endif()

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
add_library(scilib::scilib ALIAS scilib)
target_link_libraries(
    scilib 
    PRIVATE
    std::mdspan
    ${BLAS_LIBRARIES}
)

target_include_directories(scilib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${LAPACKE_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS scilib
    EXPORT scilibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    
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

