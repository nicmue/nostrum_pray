#pragma once

#include <cstddef>

#if defined(__AVX__)
    #define ENABLE_AVX
#elif defined(__SSE3__)
    #define ENABLE_SSE
#endif

#if defined(WITH_CUDA)
    #define ENABLE_CUDA
#endif

#if defined(ENABLE_SSE) || defined(ENABLE_AVX)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

namespace pray {
    typedef enum { RAYTRACING=0, PATHTRACING=1 } Mode;
    typedef enum { NORMAL=0, DIFFUSE=1 } PtMode;
}

constexpr size_t MAX_DEPTH_DIFFUSE = 1;

constexpr size_t MIN_NUM_TRIANGLES_SAH = 20000;
constexpr size_t MAX_NUM_TRIANGLES_SAH = 1000000;
constexpr size_t MAX_NUM_TRIANGLES_GPU = 10000;
constexpr size_t SAMPLE_EVENTS_THRES = 100000000;
constexpr size_t NUM_TRIANGLES_SAMPLING = 10000;
