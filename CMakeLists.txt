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

add_executable(pd_corrosion
    src/main.cpp
    src/config.cpp
    src/grid.cpp
    src/grains.cpp
    src/pd_ns.cpp
    src/pd_ard.cpp
    src/boundary.cpp
    src/coupling.cpp
    src/vtk_writer.cpp
)

target_compile_definitions(pd_corrosion PRIVATE PD_DIM=${PD_DIM})
target_link_libraries(pd_corrosion PRIVATE OpenMP::OpenMP_CXX)

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
