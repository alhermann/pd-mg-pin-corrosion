cmake_minimum_required(VERSION 3.14)
project(PeridynamicCorrosion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dimension: 2 (r-z) or 3 (full 3D)
if(NOT DEFINED PD_DIM)
    set(PD_DIM 2)
endif()
message(STATUS "Building for PD_DIM = ${PD_DIM}")

# OpenMP — help CMake find Homebrew libomp on macOS with Apple Clang
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP REQUIRED)

# Eigen (header-only) for sparse linear algebra in implicit ARD solver
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

add_executable(pd_corrosion
    src/main.cpp
    src/config.cpp
    src/grid.cpp
    src/grains.cpp
    src/pd_ns.cpp
    src/pd_ard.cpp
    src/pd_ard_implicit.cpp
    src/boundary.cpp
    src/coupling.cpp
    src/vtk_writer.cpp
)

target_compile_definitions(pd_corrosion PRIVATE PD_DIM=${PD_DIM})
target_link_libraries(pd_corrosion PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)

# Test executable for implicit ARD solver validation
add_executable(test_implicit
    tests/test_implicit.cpp
    src/config.cpp
    src/grid.cpp
    src/pd_ard.cpp
    src/pd_ard_implicit.cpp
    src/vtk_writer.cpp
)
target_include_directories(test_implicit PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_implicit PRIVATE PD_DIM=${PD_DIM})
target_link_libraries(test_implicit PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)

# AMR validation tests
add_executable(test_amr
    tests/test_amr.cpp
    src/config.cpp
    src/grid.cpp
    src/pd_ns.cpp
    src/pd_ard.cpp
    src/pd_ard_implicit.cpp
    src/boundary.cpp
    src/vtk_writer.cpp
)
target_include_directories(test_amr PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_amr PRIVATE PD_DIM=${PD_DIM})
target_link_libraries(test_amr PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(pd_corrosion PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
)
target_link_options(pd_corrosion PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)
