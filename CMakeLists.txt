cmake_minimum_required(VERSION 3.3)
project(Pray CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(USE_GPU_ARG ON CACHE BOOL "Use GPU Acceleration")
set(NVCC_COMPILER_ARG "c++" CACHE STRING "User defined c++ compiler")
set(NVCC_ARCH_ARG "" CACHE STRING "GPU Compute Architecture")

if(NVCC_ARCH_ARG STREQUAL "")
    set(GPU_ARCH_SET OFF CACHE BOOL "GPU Arch was not given")
else()
    set(GPU_ARCH_SET ON CACHE BOOL "GPU Arch was given")
endif()


option(WITH_OMP "Build with OpenMP support" ON)
find_package(OpenMP)
if (NOT OpenMP_FOUND)
    set(WITH_OMP OFF CACHE BOOL "Build with OpenMP support" FORCE)
endif()

option(WITH_CUDA "Build with CUDA support" ON)
if(USE_GPU_ARG)
    find_package(CUDA)
    if (NOT CUDA_FOUND OR NOT CUDA_CUDA_LIBRARY)
        set(WITH_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
    endif()
else()
    set(WITH_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
endif()

macro(print_opt OPT)
    if(${OPT})
        message(STATUS "${OPT} is ON")
    else()
        message(STATUS "${OPT} is OFF")
    endif()
endmacro()

message(STATUS "")
message(STATUS "##### CONFIG #####")
print_opt(WITH_OMP)
print_opt(WITH_CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Werror=vla -Werror -march=native -Ofast -fno-fast-math")

if (WITH_CUDA)
    message(STATUS "Using ${NVCC_COMPILER_ARG} as cuda host compiler")
    if (GPU_ARCH_SET)
        message(STATUS "GPU Architecture given: ${NVCC_ARCH_ARG}")
    endif()
    message(STATUS "##################\n")

    set(CUDA_HOST_COMPILER ${NVCC_COMPILER_ARG})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11 --optimize 3 -Wno-deprecated-gpu-targets")

    if(GPU_ARCH_SET)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode ${NVCC_ARCH_ARG}")
    endif()
endif()

add_subdirectory(src)
