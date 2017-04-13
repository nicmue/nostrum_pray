#include "geometry_simd.h"

namespace pray {

#if defined(ENABLE_AVX)
    __m256 intersectBB_avx(const BoundingBox &bb, const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                           const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ) {
        __m256 minX = _mm256_set1_ps(bb.min.x);
        __m256 minY = _mm256_set1_ps(bb.min.y);
        __m256 minZ = _mm256_set1_ps(bb.min.z);

        __m256 maxX = _mm256_set1_ps(bb.max.x);
        __m256 maxY = _mm256_set1_ps(bb.max.y);
        __m256 maxZ = _mm256_set1_ps(bb.max.z);

        __m256 tNear = _mm256_set1_ps(std::numeric_limits<float>::lowest());
        __m256 tFar = _mm256_set1_ps(std::numeric_limits<float>::max());

        // div instead of mul to use inverse direction
        __m256 t1X = _mm256_div_ps(_mm256_sub_ps(minX, rayPosX), rayDirX);
        __m256 t1Y = _mm256_div_ps(_mm256_sub_ps(minY, rayPosY), rayDirY);
        __m256 t1Z = _mm256_div_ps(_mm256_sub_ps(minZ, rayPosZ), rayDirZ);

        __m256 t2X = _mm256_div_ps(_mm256_sub_ps(maxX, rayPosX), rayDirX);
        __m256 t2Y = _mm256_div_ps(_mm256_sub_ps(maxY, rayPosY), rayDirY);
        __m256 t2Z = _mm256_div_ps(_mm256_sub_ps(maxZ, rayPosZ), rayDirZ);

        __m256 tNear2X = _mm256_min_ps(t1X, t2X);
        __m256 tNear2Y = _mm256_min_ps(t1Y, t2Y);
        __m256 tNear2Z = _mm256_min_ps(t1Z, t2Z);

        __m256 tFar2X = _mm256_max_ps(t1X, t2X);
        __m256 tFar2Y = _mm256_max_ps(t1Y, t2Y);
        __m256 tFar2Z = _mm256_max_ps(t1Z, t2Z);

        tNear = _mm256_max_ps(_mm256_max_ps(tNear2X, tNear2Y), _mm256_max_ps(tNear2Z, tNear));
        tFar = _mm256_min_ps(_mm256_min_ps(tFar2X, tFar2Y), _mm256_min_ps(tFar2Z, tFar));

        return _mm256_cmp_ps(tNear, tFar, _CMP_LE_OQ);
    }

    __m256 intersectTriangle_avx(const Triangle &tri, const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                                 const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ, const __m256 &rayMask,
                                 __m256 &distance) {
        __m256 abX = _mm256_set1_ps(tri.ab.x);
        __m256 abY = _mm256_set1_ps(tri.ab.y);
        __m256 abZ = _mm256_set1_ps(tri.ab.z);

        __m256 acX = _mm256_set1_ps(tri.ac.x);
        __m256 acY = _mm256_set1_ps(tri.ac.y);
        __m256 acZ = _mm256_set1_ps(tri.ac.z);

        __m256 pvecX, pvecY, pvecZ;
        cross_avx(rayDirX, rayDirY, rayDirZ, acX, acY, acZ, pvecX, pvecY, pvecZ);
        __m256 det = dot_avx(abX, abY, abZ, pvecX, pvecY, pvecZ);

        // true: 0xffffffff, false: 0x0
        __m256 result = _mm256_and_ps(rayMask, _mm256_cmp_ps(det, _mm256_set1_ps(EPSILON), _CMP_GE_OQ));
        if (_mm256_movemask_ps(result) == 0) {
            return result;
        }

        __m256 invDet = _mm256_div_ps(_mm256_set1_ps(1.f), det);

        __m256 aX = _mm256_set1_ps(tri.a.x);
        __m256 aY = _mm256_set1_ps(tri.a.y);
        __m256 aZ = _mm256_set1_ps(tri.a.z);

        __m256 tvecX = _mm256_sub_ps(rayPosX, aX);
        __m256 tvecY = _mm256_sub_ps(rayPosY, aY);
        __m256 tvecZ = _mm256_sub_ps(rayPosZ, aZ);

        __m256 pdotT = dot_avx(tvecX, tvecY, tvecZ, pvecX, pvecY, pvecZ);
        __m256 u = _mm256_mul_ps(pdotT, invDet);

        result = _mm256_and_ps(result, _mm256_cmp_ps(u, _mm256_set1_ps(0.f), _CMP_GE_OQ));
        result = _mm256_and_ps(result, _mm256_cmp_ps(u, _mm256_set1_ps(1.f), _CMP_LE_OQ));
        if (_mm256_movemask_ps(result) == 0) {
            return result;
        }

        __m256 qvecX, qvecY, qvecZ;
        cross_avx(tvecX, tvecY, tvecZ, abX, abY, abZ, qvecX, qvecY, qvecZ);

        __m256 v = _mm256_mul_ps(dot_avx(rayDirX, rayDirY, rayDirZ, qvecX, qvecY, qvecZ), invDet);

        result = _mm256_and_ps(result, _mm256_cmp_ps(v, _mm256_set1_ps(0.f), _CMP_GE_OQ));
        result = _mm256_and_ps(result, _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.f), _CMP_LE_OQ));
        if (_mm256_movemask_ps(result) == 0) {
            return result;
        }

        distance = _mm256_mul_ps(dot_avx(acX, acY, acZ, qvecX, qvecY, qvecZ), invDet);

        return result;
    }

#elif defined(ENABLE_SSE)
    __m128 intersectBB_sse(const BoundingBox &bb, const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                           const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ) {
        __m128 minX = _mm_set_ps1(bb.min.x);
        __m128 minY = _mm_set_ps1(bb.min.y);
        __m128 minZ = _mm_set_ps1(bb.min.z);

        __m128 maxX = _mm_set_ps1(bb.max.x);
        __m128 maxY = _mm_set_ps1(bb.max.y);
        __m128 maxZ = _mm_set_ps1(bb.max.z);

        __m128 tNear = _mm_set_ps1(std::numeric_limits<float>::lowest());
        __m128 tFar = _mm_set_ps1(std::numeric_limits<float>::max());

        // div instead of mul to use inverse direction
        __m128 t1X = _mm_div_ps(_mm_sub_ps(minX, rayPosX), rayDirX);
        __m128 t1Y = _mm_div_ps(_mm_sub_ps(minY, rayPosY), rayDirY);
        __m128 t1Z = _mm_div_ps(_mm_sub_ps(minZ, rayPosZ), rayDirZ);

        __m128 t2X = _mm_div_ps(_mm_sub_ps(maxX, rayPosX), rayDirX);
        __m128 t2Y = _mm_div_ps(_mm_sub_ps(maxY, rayPosY), rayDirY);
        __m128 t2Z = _mm_div_ps(_mm_sub_ps(maxZ, rayPosZ), rayDirZ);

        __m128 tNear2X = _mm_min_ps(t1X, t2X);
        __m128 tNear2Y = _mm_min_ps(t1Y, t2Y);
        __m128 tNear2Z = _mm_min_ps(t1Z, t2Z);

        __m128 tFar2X = _mm_max_ps(t1X, t2X);
        __m128 tFar2Y = _mm_max_ps(t1Y, t2Y);
        __m128 tFar2Z = _mm_max_ps(t1Z, t2Z);

        tNear = _mm_max_ps(_mm_max_ps(tNear2X, tNear2Y), _mm_max_ps(tNear2Z, tNear));
        tFar = _mm_min_ps(_mm_min_ps(tFar2X, tFar2Y), _mm_min_ps(tFar2Z, tFar));

        return _mm_cmple_ps(tNear, tFar);
    }

    // Moeller-Trumbore algorithm implemented in sse
    __m128 intersectTriangle_sse(const Triangle &tri, const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                                 const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ, const __m128 &rayMask,
                                 __m128 &distance) {
        __m128 abX = _mm_set_ps1(tri.ab.x);
        __m128 abY = _mm_set_ps1(tri.ab.y);
        __m128 abZ = _mm_set_ps1(tri.ab.z);

        __m128 acX = _mm_set_ps1(tri.ac.x);
        __m128 acY = _mm_set_ps1(tri.ac.y);
        __m128 acZ = _mm_set_ps1(tri.ac.z);

        __m128 pvecX, pvecY, pvecZ;
        cross_sse(rayDirX, rayDirY, rayDirZ, acX, acY, acZ, pvecX, pvecY, pvecZ);
        __m128 det = dot_sse(abX, abY, abZ, pvecX, pvecY, pvecZ);

        // true: 0xffffffff, false: 0x0
        __m128 result = _mm_and_ps(rayMask, _mm_cmpge_ps(det, _mm_set_ps1(EPSILON)));
        if (_mm_movemask_ps(result) == 0) {
            return result;
        }

        __m128 invDet = _mm_div_ps(_mm_set_ps1(1.f), det);

        __m128 aX = _mm_set_ps1(tri.a.x);
        __m128 aY = _mm_set_ps1(tri.a.y);
        __m128 aZ = _mm_set_ps1(tri.a.z);

        __m128 tvecX = _mm_sub_ps(rayPosX, aX);
        __m128 tvecY = _mm_sub_ps(rayPosY, aY);
        __m128 tvecZ = _mm_sub_ps(rayPosZ, aZ);

        __m128 pdotT = dot_sse(tvecX, tvecY, tvecZ, pvecX, pvecY, pvecZ);
        __m128 u = _mm_mul_ps(pdotT, invDet);

        result = _mm_and_ps(result, _mm_cmpge_ps(u, _mm_set_ps1(0.f)));
        result = _mm_and_ps(result, _mm_cmple_ps(u, _mm_set_ps1(1.f)));
        if (_mm_movemask_ps(result) == 0) {
            return result;
        }

        __m128 qvecX, qvecY, qvecZ;
        cross_sse(tvecX, tvecY, tvecZ, abX, abY, abZ, qvecX, qvecY, qvecZ);

        __m128 v = _mm_mul_ps(dot_sse(rayDirX, rayDirY, rayDirZ, qvecX, qvecY, qvecZ), invDet);

        result = _mm_and_ps(result, _mm_cmpge_ps(v, _mm_set_ps1(0.f)));
        result = _mm_and_ps(result, _mm_cmple_ps(_mm_add_ps(u, v), _mm_set_ps1(1.f)));
        if (_mm_movemask_ps(result) == 0) {
            return result;
        }

        distance = _mm_mul_ps(dot_sse(acX, acY, acZ, qvecX, qvecY, qvecZ), invDet);

        return result;
    }
#endif
}