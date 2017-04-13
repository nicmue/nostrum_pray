#include "pray.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <geometry.h>
#include <iostream>
#include <chrono>

#include "geometry_simd.h"

#ifdef ENABLE_CUDA
    #include "path_trace.cu.h"
#endif

namespace pray {
    Vec3 evaluateRay(size_t rayIdx, const Scene &scene) {
        const Ray &ray = scene.primRays[rayIdx];

        Triangle intersectedTriangle;
        float distance;
        if (!scene.triangleDB->intersect(ray, intersectedTriangle, distance)) {
            return scene.backgroundColor;
        }

        Vec3 color(0.f, 0.f, 0.f);
        Vec3 intersectionPos = ray.position + ray.direction * distance;
        for (auto &l : scene.lights) {
            Vec3 L = l.position - intersectionPos;
            float lightDistSq = L.norm();
            L.normalize();

            // Optimization for many lights (light is behind triangle, angle > 90 degrees)
            float lambert = L.dot(intersectedTriangle.normal);
            if (lambert < 0.f) {
                continue;
            }

            // Intersect shadow ray
            Triangle tri; // not needed
            float dst;
            if (scene.triangleDB->intersect(Ray(intersectionPos, L), tri, dst) && lightDistSq > dst * dst) {
                continue;
            }

            // Shading
            color = color + (intersectedTriangle.color * l.color * lambert * (1 / lightDistSq));

            // Stop when pixel color is already white
            if (color.x >= 1.f && color.y >= 1.f && color.z >= 1.f) {
                break;
            }
        }
        return color.clamp(0.f, 1.f);
    }

#if defined(ENABLE_AVX)
    std::vector<Vec3> evaluateRay(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                               const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ,
                               const Scene &scene) {
        std::array<Triangle, 8> intersectedTriangles;
        __m256 distance;
        __m256 hitMask = scene.triangleDB->intersect(rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                                     _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff)), intersectedTriangles, distance);

        if (_mm256_movemask_ps(hitMask) == 0x0) {
            // None of the rays hit
            return std::vector<Vec3>{scene.backgroundColor, scene.backgroundColor, scene.backgroundColor, scene.backgroundColor,
                     scene.backgroundColor, scene.backgroundColor, scene.backgroundColor, scene.backgroundColor};
        }

        __m256 colorR = _mm256_blendv_ps(_mm256_set1_ps(scene.backgroundColor.x), _mm256_setzero_ps(), hitMask);
        __m256 colorG = _mm256_blendv_ps(_mm256_set1_ps(scene.backgroundColor.y), _mm256_setzero_ps(), hitMask);
        __m256 colorB = _mm256_blendv_ps(_mm256_set1_ps(scene.backgroundColor.z), _mm256_setzero_ps(), hitMask);

        __m256 intersectionPosX = _mm256_add_ps(rayPosX, _mm256_mul_ps(rayDirX, distance));
        __m256 intersectionPosY = _mm256_add_ps(rayPosY, _mm256_mul_ps(rayDirY, distance));
        __m256 intersectionPosZ = _mm256_add_ps(rayPosZ, _mm256_mul_ps(rayDirZ, distance));

        for (auto &l : scene.lights) {
            __m256 LX = _mm256_sub_ps(_mm256_set1_ps(l.position.x), intersectionPosX);
            __m256 LY = _mm256_sub_ps(_mm256_set1_ps(l.position.y), intersectionPosY);
            __m256 LZ = _mm256_sub_ps(_mm256_set1_ps(l.position.z), intersectionPosZ);

            __m256 lightDistSq = norm_avx(LX, LY, LZ);
            normalize_avx(LX, LY, LZ);

            // Optimization for many lights (light is behind triangle, angle > 90 degrees)
            __m256 itnX = _mm256_set_ps(intersectedTriangles[7].normal.x,
                                        intersectedTriangles[6].normal.x,
                                        intersectedTriangles[5].normal.x,
                                        intersectedTriangles[4].normal.x,
                                        intersectedTriangles[3].normal.x,
                                        intersectedTriangles[2].normal.x,
                                        intersectedTriangles[1].normal.x,
                                        intersectedTriangles[0].normal.x);

            __m256 itnY = _mm256_set_ps(intersectedTriangles[7].normal.y,
                                        intersectedTriangles[6].normal.y,
                                        intersectedTriangles[5].normal.y,
                                        intersectedTriangles[4].normal.y,
                                        intersectedTriangles[3].normal.y,
                                        intersectedTriangles[2].normal.y,
                                        intersectedTriangles[1].normal.y,
                                        intersectedTriangles[0].normal.y);

            __m256 itnZ = _mm256_set_ps(intersectedTriangles[7].normal.z,
                                        intersectedTriangles[6].normal.z,
                                        intersectedTriangles[5].normal.z,
                                        intersectedTriangles[4].normal.z,
                                        intersectedTriangles[3].normal.z,
                                        intersectedTriangles[2].normal.z,
                                        intersectedTriangles[1].normal.z,
                                        intersectedTriangles[0].normal.z);

            __m256 lambert = dot_avx(LX, LY, LZ, itnX, itnY, itnZ);
            __m256 shadowRayMask = _mm256_and_ps(hitMask, _mm256_cmp_ps(lambert, _mm256_setzero_ps(), _CMP_GE_OQ));
            if (_mm256_movemask_ps(shadowRayMask) == 0)
            {
                continue;
            }

            // Intersect shadow ray
            std::array<Triangle, 8> tris; // not needed
            __m256 shadowDistance;
            __m256 shadowHit = scene.triangleDB->intersect(intersectionPosX, intersectionPosY, intersectionPosZ, LX, LY, LZ,
                                                           shadowRayMask, tris, shadowDistance);

            shadowRayMask = _mm256_andnot_ps(_mm256_and_ps(shadowHit, _mm256_cmp_ps(lightDistSq, _mm256_mul_ps(shadowDistance, shadowDistance), _CMP_GE_OQ)), hitMask);
            if (_mm256_movemask_ps(shadowRayMask) == 0)
            {
                continue;
            }

            // Shading
            __m256 itcX = _mm256_set_ps(intersectedTriangles[7].color.x,
                                        intersectedTriangles[6].color.x,
                                        intersectedTriangles[5].color.x,
                                        intersectedTriangles[4].color.x,
                                        intersectedTriangles[3].color.x,
                                        intersectedTriangles[2].color.x,
                                        intersectedTriangles[1].color.x,
                                        intersectedTriangles[0].color.x);

            __m256 itcY = _mm256_set_ps(intersectedTriangles[7].color.y,
                                        intersectedTriangles[6].color.y,
                                        intersectedTriangles[5].color.y,
                                        intersectedTriangles[4].color.y,
                                        intersectedTriangles[3].color.y,
                                        intersectedTriangles[2].color.y,
                                        intersectedTriangles[1].color.y,
                                        intersectedTriangles[0].color.y);

            __m256 itcZ = _mm256_set_ps(intersectedTriangles[7].color.z,
                                        intersectedTriangles[6].color.z,
                                        intersectedTriangles[5].color.z,
                                        intersectedTriangles[4].color.z,
                                        intersectedTriangles[3].color.z,
                                        intersectedTriangles[2].color.z,
                                        intersectedTriangles[1].color.z,
                                        intersectedTriangles[0].color.z);

            __m256 lambertDecay = _mm256_div_ps(lambert, lightDistSq);
            colorR = _mm256_add_ps(colorR,
                                _mm256_and_ps(shadowRayMask, _mm256_mul_ps(itcX, _mm256_mul_ps(_mm256_set1_ps(l.color.x), lambertDecay))));
            colorG = _mm256_add_ps(colorG,
                                _mm256_and_ps(shadowRayMask, _mm256_mul_ps(itcY, _mm256_mul_ps(_mm256_set1_ps(l.color.y), lambertDecay))));
            colorB = _mm256_add_ps(colorB,
                                _mm256_and_ps(shadowRayMask, _mm256_mul_ps(itcZ, _mm256_mul_ps(_mm256_set1_ps(l.color.z), lambertDecay))));

            // Stop when pixel color is already white
            __m256 const1f = _mm256_set1_ps(1.f);
            __m256 whitePxMask = _mm256_and_ps(_mm256_cmp_ps(colorR, const1f, _CMP_GE_OQ),
                                            _mm256_and_ps(_mm256_cmp_ps(colorG, const1f, _CMP_GE_OQ),
                                                       _mm256_cmp_ps(colorB, const1f, _CMP_GE_OQ)));
            if (_mm256_movemask_ps(whitePxMask) == 0xff) {
                break;
            }
        }

        alignas(32) std::array<float, 8> cR;
        alignas(32) std::array<float, 8> cG;
        alignas(32) std::array<float, 8> cB;
        _mm256_store_ps(cR.data(), colorR);
        _mm256_store_ps(cG.data(), colorG);
        _mm256_store_ps(cB.data(), colorB);

        return std::vector<Vec3>{Vec3(cR[0], cG[0], cB[0]).clamp(0.f, 1.f),
                                 Vec3(cR[1], cG[1], cB[1]).clamp(0.f, 1.f),
                                 Vec3(cR[2], cG[2], cB[2]).clamp(0.f, 1.f),
                                 Vec3(cR[3], cG[3], cB[3]).clamp(0.f, 1.f),
                                 Vec3(cR[4], cG[4], cB[4]).clamp(0.f, 1.f),
                                 Vec3(cR[5], cG[5], cB[5]).clamp(0.f, 1.f),
                                 Vec3(cR[6], cG[6], cB[6]).clamp(0.f, 1.f),
                                 Vec3(cR[7], cG[7], cB[7]).clamp(0.f, 1.f)};
    }

#elif defined(ENABLE_SSE)
    std::vector<Vec3> evaluateRay(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                               const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ,
                               const Scene &scene) {
        std::vector<Triangle> intersectedTriangles(4);
        __m128 distance;
        __m128 hitMask = scene.triangleDB->intersect(rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                                 _mm_castsi128_ps(_mm_set1_epi32(0xffffffff)), intersectedTriangles, distance);

        if (_mm_movemask_ps(hitMask) == 0x0) {
            // None of the rays hit
            return std::vector<Vec3>(
                    {scene.backgroundColor, scene.backgroundColor, scene.backgroundColor, scene.backgroundColor});
        }

        __m128 colorR1 = _mm_andnot_ps(hitMask, _mm_set1_ps(scene.backgroundColor.x));
        __m128 colorR = _mm_add_ps(colorR1, _mm_setzero_ps());

        __m128 colorG1 = _mm_andnot_ps(hitMask, _mm_set1_ps(scene.backgroundColor.y));
        __m128 colorG = _mm_add_ps(colorG1, _mm_setzero_ps());

        __m128 colorB1 = _mm_andnot_ps(hitMask, _mm_set1_ps(scene.backgroundColor.z));
        __m128 colorB = _mm_add_ps(colorB1, _mm_setzero_ps());

        __m128 intersectionPosX = _mm_add_ps(rayPosX, _mm_mul_ps(rayDirX, distance));
        __m128 intersectionPosY = _mm_add_ps(rayPosY, _mm_mul_ps(rayDirY, distance));
        __m128 intersectionPosZ = _mm_add_ps(rayPosZ, _mm_mul_ps(rayDirZ, distance));

        for (auto &l : scene.lights) {
            __m128 LX = _mm_sub_ps(_mm_set_ps1(l.position.x), intersectionPosX);
            __m128 LY = _mm_sub_ps(_mm_set_ps1(l.position.y), intersectionPosY);
            __m128 LZ = _mm_sub_ps(_mm_set_ps1(l.position.z), intersectionPosZ);

            __m128 lightDistSq = norm_sse(LX, LY, LZ);
            normalize_sse(LX, LY, LZ);

            // Optimization for many lights (light is behind triangle, angle > 90 degrees)
            __m128 itnX = _mm_set_ps(intersectedTriangles[3].normal.x,
                                     intersectedTriangles[2].normal.x,
                                     intersectedTriangles[1].normal.x,
                                     intersectedTriangles[0].normal.x);

            __m128 itnY = _mm_set_ps(intersectedTriangles[3].normal.y,
                                     intersectedTriangles[2].normal.y,
                                     intersectedTriangles[1].normal.y,
                                     intersectedTriangles[0].normal.y);

            __m128 itnZ = _mm_set_ps(intersectedTriangles[3].normal.z,
                                     intersectedTriangles[2].normal.z,
                                     intersectedTriangles[1].normal.z,
                                     intersectedTriangles[0].normal.z);

            __m128 lambert = dot_sse(LX, LY, LZ, itnX, itnY, itnZ);
            __m128 shadowRayMask = _mm_and_ps(hitMask, _mm_cmpge_ps(lambert, _mm_setzero_ps()));
            if (_mm_movemask_ps(shadowRayMask) == 0)
            {
                continue;
            }

            // Intersect shadow ray
            std::vector<Triangle> tris(4); // not needed
            __m128 shadowDistance;
            __m128 shadowHit = scene.triangleDB->intersect(intersectionPosX, intersectionPosY, intersectionPosZ, LX, LY, LZ,
                                                           shadowRayMask, tris, shadowDistance);

            shadowRayMask = _mm_andnot_ps(_mm_and_ps(shadowHit, _mm_cmpge_ps(lightDistSq, _mm_mul_ps(shadowDistance, shadowDistance))), hitMask);
            if (_mm_movemask_ps(shadowRayMask) == 0)
            {
                continue;
            }

            // Shading
            __m128 itcX = _mm_set_ps(intersectedTriangles[3].color.x,
                                     intersectedTriangles[2].color.x,
                                     intersectedTriangles[1].color.x,
                                     intersectedTriangles[0].color.x);

            __m128 itcY = _mm_set_ps(intersectedTriangles[3].color.y,
                                     intersectedTriangles[2].color.y,
                                     intersectedTriangles[1].color.y,
                                     intersectedTriangles[0].color.y);

            __m128 itcZ = _mm_set_ps(intersectedTriangles[3].color.z,
                                     intersectedTriangles[2].color.z,
                                     intersectedTriangles[1].color.z,
                                     intersectedTriangles[0].color.z);

            __m128 lambertDecay = _mm_div_ps(lambert, lightDistSq);
            colorR = _mm_add_ps(colorR,
                                _mm_and_ps(shadowRayMask, _mm_mul_ps(itcX, _mm_mul_ps(_mm_set_ps1(l.color.x), lambertDecay))));
            colorG = _mm_add_ps(colorG,
                                _mm_and_ps(shadowRayMask, _mm_mul_ps(itcY, _mm_mul_ps(_mm_set_ps1(l.color.y), lambertDecay))));
            colorB = _mm_add_ps(colorB,
                                _mm_and_ps(shadowRayMask, _mm_mul_ps(itcZ, _mm_mul_ps(_mm_set_ps1(l.color.z), lambertDecay))));

            // Stop when pixel color is already white
            __m128 const1f = _mm_set_ps1(1.f);
            __m128 whitePxMask = _mm_and_ps(_mm_cmpge_ps(colorR, const1f),
                                            _mm_and_ps(_mm_cmpge_ps(colorG, const1f),
                                                       _mm_cmpge_ps(colorB, const1f)));
            if (_mm_movemask_ps(whitePxMask) == 0xf) {
                break;
            }
        }

        alignas(16) std::array<float, 4> cR, cG, cB;
        _mm_store_ps(cR.data(), colorR);
        _mm_store_ps(cG.data(), colorG);
        _mm_store_ps(cB.data(), colorB);

        return std::vector<Vec3>{Vec3(cR[0], cG[0], cB[0]).clamp(0.f, 1.f),
                                 Vec3(cR[1], cG[1], cB[1]).clamp(0.f, 1.f),
                                 Vec3(cR[2], cG[2], cB[2]).clamp(0.f, 1.f),
                                 Vec3(cR[3], cG[3], cB[3]).clamp(0.f, 1.f)};
    }
#endif

    Vec3 evaluatePathRecursiveDiffuse(const Ray &ray, const Scene &scene, size_t depth) {
        Triangle intersectedTriangle;
        float distance;
        if (!scene.triangleDB->intersect(ray, intersectedTriangle, distance)) {
            return scene.backgroundColor;
        }

        if (intersectedTriangle.isEmitting) {
            return intersectedTriangle.color;
        } else if (depth >= scene.maxDepth) {
            return Vec3(0.f, 0.f, 0.f);
        }

        Vec3 intersectionPos = ray.position + ray.direction * distance;
        Vec3 color(0.f, 0.f, 0.f);

        std::mt19937_64 rng(static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()));
        std::uniform_real_distribution<float> uniform;

        for (size_t i = 0; i < scene.numSamples; i++) {
            Ray newRay(intersectionPos, sampleHemisphere(intersectedTriangle.normal, uniform(rng), uniform(rng)));
            color = color + evaluatePathRecursiveDiffuse(newRay, scene, depth + 1);
        }

        return (color * intersectedTriangle.color) / scene.numSamples;
    }

    Vec3 evaluatePathRecursive(const Ray &ray, const Scene &scene, size_t depth) {
        Triangle intersectedTriangle;
        float distance;
        if (!scene.triangleDB->intersect(ray, intersectedTriangle, distance)) {
            return scene.backgroundColor;
        }

        if (intersectedTriangle.isEmitting) {
            return intersectedTriangle.color;
        } else if (depth >= scene.maxDepth) {
            return Vec3(0.f, 0.f, 0.f);
        }

        Vec3 intersectionPos = ray.position + ray.direction * distance;

        std::mt19937_64 rng(static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()));
        std::uniform_real_distribution<float> uniform;

        Ray newRay(intersectionPos, sampleHemisphere(intersectedTriangle.normal, uniform(rng), uniform(rng)));
        Vec3 color = evaluatePathRecursive(newRay, scene, depth + 1);

        return color * intersectedTriangle.color;
    }

    Vec3 evaluatePath(size_t rayIdx, const Scene &scene) {
        Vec3 result(0.f, 0.f, 0.f);
        if (scene.ptMode == PtMode::NORMAL) {
            for (uint64_t i = 0; i < scene.numSamples; i++) {
                result += evaluatePathRecursive(scene.primRays[rayIdx], scene, 0);
            }
            result = (result / scene.numSamples);
        } else {
            result = evaluatePathRecursiveDiffuse(scene.primRays[rayIdx], scene, 0);
        }

        return result.clamp(0.f, 1.f);
    }

    void rayTrace(Image &image, const Scene &scene)
    {
        uint64_t rowLen = scene.width * 3;

#if defined(ENABLE_AVX)
        // for each pixel of the image
        #pragma omp parallel for schedule(dynamic, 10) collapse(2)
        for (uint64_t yPixel = 0; yPixel <= scene.height - 2; yPixel += 2) {
            for (uint64_t xPixel = 0; xPixel <= scene.width - 4; xPixel += 4) {

                const Ray &primRay0 = scene.primRays[xPixel + yPixel * scene.width];
                const Ray &primRay1 = scene.primRays[xPixel + 1 + yPixel * scene.width];
                const Ray &primRay2 = scene.primRays[xPixel + 2 + yPixel * scene.width];
                const Ray &primRay3 = scene.primRays[xPixel + 3 + yPixel * scene.width];
                const Ray &primRay4 = scene.primRays[xPixel + (yPixel + 1) * scene.width];
                const Ray &primRay5 = scene.primRays[xPixel + 1 + (yPixel + 1) * scene.width];
                const Ray &primRay6 = scene.primRays[xPixel + 2 + (yPixel + 1) * scene.width];
                const Ray &primRay7 = scene.primRays[xPixel + 3 + (yPixel + 1) * scene.width];

                __m256 rayPosX = _mm256_set_ps(primRay7.position.x, primRay6.position.x, primRay5.position.x,
                                               primRay4.position.x, primRay3.position.x, primRay2.position.x,
                                               primRay1.position.x, primRay0.position.x);
                __m256 rayPosY = _mm256_set_ps(primRay7.position.y, primRay6.position.y, primRay5.position.y,
                                               primRay4.position.y, primRay3.position.y, primRay2.position.y,
                                               primRay1.position.y, primRay0.position.y);
                __m256 rayPosZ = _mm256_set_ps(primRay7.position.z, primRay6.position.z, primRay5.position.z,
                                               primRay4.position.z, primRay3.position.z, primRay2.position.z,
                                               primRay1.position.z, primRay0.position.z);

                __m256 rayDirX = _mm256_set_ps(primRay7.direction.x, primRay6.direction.x, primRay5.direction.x,
                                               primRay4.direction.x, primRay3.direction.x, primRay2.direction.x,
                                               primRay1.direction.x, primRay0.direction.x);
                __m256 rayDirY = _mm256_set_ps(primRay7.direction.y, primRay6.direction.y, primRay5.direction.y,
                                               primRay4.direction.y, primRay3.direction.y, primRay2.direction.y,
                                               primRay1.direction.y, primRay0.direction.y);
                __m256 rayDirZ = _mm256_set_ps(primRay7.direction.z, primRay6.direction.z, primRay5.direction.z,
                                               primRay4.direction.z, primRay3.direction.z, primRay2.direction.z,
                                               primRay1.direction.z, primRay0.direction.z);

                auto colors = evaluateRay(rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, scene);

                // scale and round to nearest
                for (auto &color : colors)
                {
                    color = (color * 255.f) + 0.5f;
                }

                uint64_t xPos = xPixel * 3;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(colors[0].x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[0].y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[0].z);

                image[(xPos + 3) + yPixel * rowLen] = static_cast<uint8_t>(colors[1].x);
                image[(xPos + 3) + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[1].y);
                image[(xPos + 3) + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[1].z);

                image[(xPos + 6) + yPixel * rowLen] = static_cast<uint8_t>(colors[2].x);
                image[(xPos + 6) + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[2].y);
                image[(xPos + 6) + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[2].z);

                image[(xPos + 9) + yPixel * rowLen] = static_cast<uint8_t>(colors[3].x);
                image[(xPos + 9) + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[3].y);
                image[(xPos + 9) + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[3].z);

                image[xPos + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[4].x);
                image[xPos + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[4].y);
                image[xPos + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[4].z);

                image[(xPos + 3) + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[5].x);
                image[(xPos + 3) + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[5].y);
                image[(xPos + 3) + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[5].z);

                image[(xPos + 6) + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[6].x);
                image[(xPos + 6) + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[6].y);
                image[(xPos + 6) + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[6].z);

                image[(xPos + 9) + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[7].x);
                image[(xPos + 9) + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[7].y);
                image[(xPos + 9) + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[7].z);
            }
        }

        // Remainder
        uint64_t missing = scene.width % 4;

        //#pragma omp parallel for schedule(dynamic, 10)
        for (uint64_t yPixel = 0; yPixel < scene.height - 1; yPixel++) {
            for (uint64_t xPixel = scene.width - 1 - missing; xPixel < scene.width; xPixel++) {

                Vec3 color = evaluateRay(xPixel + yPixel * scene.width, scene);

                // scale and round to nearest
                color = (color * 255.f) + 0.5f;
                uint64_t xPos = xPixel * 3;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }

        if (scene.height % 2 != 0)
        {
            uint64_t yPixel = scene.height - 1;
            for (uint64_t xPixel = 0; xPixel < scene.width; xPixel ++) {
                Vec3 color = evaluateRay(xPixel + yPixel * scene.width, scene);

                // scale and round to nearest
                color = (color * 255.f) + 0.5f;
                uint64_t xPos = xPixel * 3;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }

#elif defined(ENABLE_SSE)
        // for each pixel of the image
        #pragma omp parallel for schedule(dynamic, 10) collapse(2)
        for (uint64_t yPixel = 0; yPixel <= scene.height - 2; yPixel += 2) {
            for (uint64_t xPixel = 0; xPixel <= scene.width - 2; xPixel += 2) {
                const Ray &primRay0 = scene.primRays[xPixel + yPixel * scene.width];
                const Ray &primRay1 = scene.primRays[xPixel + 1 + yPixel * scene.width];
                const Ray &primRay2 = scene.primRays[xPixel + (yPixel + 1) * scene.width];
                const Ray &primRay3 = scene.primRays[xPixel + 1 + (yPixel + 1) * scene.width];

                __m128 rayPosX = _mm_set_ps(primRay3.position.x, primRay2.position.x, primRay1.position.x, primRay0.position.x);
                __m128 rayPosY = _mm_set_ps(primRay3.position.y, primRay2.position.y, primRay1.position.y, primRay0.position.y);
                __m128 rayPosZ = _mm_set_ps(primRay3.position.z, primRay2.position.z, primRay1.position.z, primRay0.position.z);

                __m128 rayDirX = _mm_set_ps(primRay3.direction.x, primRay2.direction.x, primRay1.direction.x, primRay0.direction.x);
                __m128 rayDirY = _mm_set_ps(primRay3.direction.y, primRay2.direction.y, primRay1.direction.y, primRay0.direction.y);
                __m128 rayDirZ = _mm_set_ps(primRay3.direction.z, primRay2.direction.z, primRay1.direction.z, primRay0.direction.z);

                auto colors = evaluateRay(rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, scene);

                // scale and round to nearest
                for (auto &color : colors)
                {
                    color = (color * 255.f) + 0.5f;
                }

                uint64_t xPos = xPixel * 3;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(colors[0].x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[0].y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[0].z);

                image[(xPos + 3) + yPixel * rowLen] = static_cast<uint8_t>(colors[1].x);
                image[(xPos + 3) + yPixel * rowLen + 1] = static_cast<uint8_t>(colors[1].y);
                image[(xPos + 3) + yPixel * rowLen + 2] = static_cast<uint8_t>(colors[1].z);

                image[xPos + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[2].x);
                image[xPos + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[2].y);
                image[xPos + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[2].z);

                image[(xPos + 3) + (yPixel + 1) * rowLen] = static_cast<uint8_t>(colors[3].x);
                image[(xPos + 3) + (yPixel + 1) * rowLen + 1] = static_cast<uint8_t>(colors[3].y);
                image[(xPos + 3) + (yPixel + 1) * rowLen + 2] = static_cast<uint8_t>(colors[3].z);
            }
        }

        // Remainder
        if (scene.width % 2 != 0)
        {
            uint64_t xPixel = scene.width - 1;
            uint64_t xPos = xPixel * 3;
            for (uint64_t yPixel = 0; yPixel < scene.height - 1; yPixel ++) {

                Vec3 color = evaluateRay(xPixel + yPixel * scene.width, scene);

                // scale and round to nearest
                color = (color * 255.f) + 0.5f;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }

        if (scene.height % 2 != 0)
        {
            uint64_t yPixel = scene.height - 1;
            for (uint64_t xPixel = 0; xPixel < scene.width; xPixel ++) {

                Vec3 color = evaluateRay(xPixel + yPixel * scene.width, scene);

                // scale and round to nearest
                uint64_t xPos = xPixel * 3;
                color = (color * 255.f) + 0.5f;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }
#else
        // for each pixel of the image
        #pragma omp parallel for schedule(dynamic, 10) collapse(2)
        for (uint64_t yPixel = 0; yPixel < scene.height; yPixel++) {
            for (uint64_t xPixel = 0; xPixel < scene.width; xPixel++) {
                uint64_t xPos = xPixel * 3;

                Vec3 color = evaluateRay(xPixel + yPixel * scene.width, scene);

                // scale and round to nearest
                color = (color * 255.f) + 0.5f;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }
#endif

    }

    void pathTrace(Image &image, const Scene &scene)
    {
#ifdef ENABLE_CUDA
        if (scene.useGPU) {
            pathTraceOnGPURecursive(image, scene);
            return;
        }
#endif

        uint64_t rowLen = scene.width * 3;
        // for each pixel of the image
        #pragma omp parallel for schedule(dynamic, 10) collapse(2)
        for (uint64_t yPixel = 0; yPixel < scene.height; yPixel++) {
            for (uint64_t xPixel = 0; xPixel < scene.width; xPixel++) {
                uint64_t xPos = xPixel * 3;
                Vec3 color = evaluatePath(xPixel + yPixel * scene.width, scene);
                // scale and round to nearest
                color = (color * 255.f) + 0.5f;
                image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
            }
        }
    }

    void render(Image &image, const Scene &scene) {
        if (scene.triangleDB->isEmpty()) {
            uint64_t rowLen = scene.width * 3;
            auto color = scene.backgroundColor * 255.f;
            color = color + 0.5f; // 0.5f because of rounding (round to nearest)
            for (uint64_t yPixel = 0; yPixel < scene.height; yPixel++) {
                for (uint64_t xPos = 0; xPos < scene.width * 3; xPos += 3) {
                    image[xPos + yPixel * rowLen] = static_cast<uint8_t>(color.x);
                    image[xPos + yPixel * rowLen + 1] = static_cast<uint8_t>(color.y);
                    image[xPos + yPixel * rowLen + 2] = static_cast<uint8_t>(color.z);
                }
            }
            return;
        }

        if (scene.mode == Mode::PATHTRACING) {
            pathTrace(image, scene);
        } else {
            rayTrace(image, scene);
        }
    }
}
