#include "path_trace.cu.h"

#include "gpu_util.h"

#include <iostream>
#include <curand_kernel.h>
#include <math_constants.h>

#define CUDA_ERR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void printGPUMemoryConsumption(int line=0){
    size_t mem_free, mem_tot;
    cudaMemGetInfo  (&mem_free, & mem_tot);
    std::cout << "Device has " << (float)mem_free/(1024*1024*1024) << " GB free of total " << (float)mem_tot/(1024*1024*1024) << " GB (Line: " << line << ")\n";
}


namespace pray {

namespace cuda {
        static constexpr float EPSILON = 1.e-4f;
        static constexpr float INFTY = std::numeric_limits<float>::max();
        static constexpr float NEG_INFTY = std::numeric_limits<float>::lowest();

        __device__ __host__ inline float clamp(float lo, float hi, const float &v) { return fmaxf(lo, fminf(hi, v)); }

        struct Vec3 {
            float x;
            float y;
            float z;

            __device__ __host__ Vec3() = default;

            __device__ __host__ Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

            __device__ __host__ Vec3(const float v[3]) : x(v[0]), y(v[1]), z(v[2]) {}

            __device__ __host__ void normalize() {
                float nor2 = norm();
                if (nor2 > 0) {
                    float invNor = 1 / std::sqrt(nor2);
                    x *= invNor, y *= invNor, z *= invNor;
                }
            }

            __device__ __host__ float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

            __device__ __host__ Vec3 cross(const Vec3 &v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

            __device__ __host__ Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }

            __device__ __host__ Vec3 operator+=(const Vec3 &v) {
                x += v.x, y += v.y, z += v.z;
                return *this;
            }

            __device__ __host__ Vec3 operator+(const float &r) const { return Vec3(x + r, y + r, z + r); }

            __device__ __host__ Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }

            __device__ __host__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

            __device__ __host__ Vec3 operator*(const float &r) const { return Vec3(x * r, y * r, z * r); }

            __device__ __host__ Vec3 operator*(const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

            __device__ __host__ Vec3 operator/(const float &r) const { return Vec3(x / r, y / r, z / r); }

            __device__ __host__ bool operator==(const Vec3 &v) const { return v.x == x && v.y == y && v.z == z; }

            __device__ __host__ const float &operator[](int i) const {
                assert(i >= 0 && i < 3);
                return (&x)[i];
            }

            __device__ __host__ float norm() const { return x * x + y * y + z * z; }

            __device__ __host__ float length() const { return std::sqrt(norm()); }

            __device__ __host__ Vec3 clamp(float lo, float hi) {
                return Vec3(pray::cuda::clamp(lo, hi, x), pray::cuda::clamp(lo, hi, y), pray::cuda::clamp(lo, hi, z));
            }

            __device__ __host__ Vec3 inverse() {
                return Vec3(1 / x, 1 / y, 1 / z);
            }
        };


        struct Ray {

            Vec3 position;
            Vec3 direction;
            Vec3 inverseDirection;

            __device__ __host__ Ray() = default;

            __device__ __host__ Ray(Vec3 _position, Vec3 _direction) : position(_position), direction(_direction) {
                direction.normalize();
                inverseDirection = direction.inverse();
            }
        };

        __device__ __host__ Vec3 sampleHemisphere(Vec3 normal, float u1, float u2) {
            float theta = acosf(sqrtf(1 - u1));
            float phi = 2.f * CUDART_PI_F * u2;

            Vec3 s(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));

            Vec3 h = normal;
            if (fabsf(h.x) <= fabsf(h.y) && fabsf(h.x) <= fabsf(h.z)) {
                h.x = 1.f;
            } else if (fabsf(h.y) <= fabsf(h.x) && fabsf(h.y) <= fabsf(h.z)) {
                h.y = 1.f;
            } else {
                h.z = 1.f;
            }


            Vec3 x = h.cross(normal);
            x.normalize();

            Vec3 z = x.cross(normal);
            z.normalize();

            Vec3 direction = x * s.x + normal * s.y + z * s.z;
            direction.normalize();
            return direction;
        }

        struct BoundingBox {
            Vec3 min;
            Vec3 max;

            __device__ __host__ BoundingBox() = default;

            __device__ __host__ ~BoundingBox() = default;

            __device__ __host__ BoundingBox(Vec3 _min, Vec3 _max) : min(_min), max(_max) {}

            __device__ __host__ bool intersect(const Ray &ray) const {

                //printf("BB: min(%f, %f, %f) max(%f, %f, %f)\n", min.x, min.y, min.z, max.x, max.y, max.z);
                //printf("%f, %f, %f in %f, %f, %f\n", ray.position.x, ray.position.y, ray.position.z, ray.direction.x, ray.direction.y, ray.direction.z);

                float tNear = NEG_INFTY;
                float tFar = INFTY;

                return  intersect(ray, tNear, tFar);
            }

            __device__ __host__ bool intersect(const Ray &ray, float &tNear, float &tFar) const {
                tNear = NEG_INFTY;
                tFar = INFTY;

                Vec3 t1 = (min - ray.position) * ray.inverseDirection;
                Vec3 t2 = (max - ray.position) * ray.inverseDirection;

                Vec3 tNear2(fminf(t1.x, t2.x), fminf(t1.y, t2.y), fminf(t1.z, t2.z));
                Vec3 tFar2(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z));

                tNear = fmaxf(fmaxf(tNear2.x, tNear2.y), fmaxf(tNear2.z, tNear));
                tFar = fminf(fminf(tFar2.x, tFar2.y), fminf(tFar2.z, tFar));

                return tNear <= tFar;
            }
        };

        struct Triangle {
            Vec3 a;
            Vec3 b;
            Vec3 c;
            Vec3 color;
            bool isEmitting;
            Vec3 normal;
            Vec3 ab; // Vec from a to b
            Vec3 ac; // Vec from a to c

            Vec3 midPoint;
            BoundingBox boundingBox;

            __device__ __host__ Triangle() = default;

            __device__ __host__ ~Triangle() = default;

            __device__ __host__ Triangle(Vec3 _a, Vec3 _b, Vec3 _c, Vec3 _color, bool _isEmitting) : a(_a), b(_b), c(_c), color(_color),
                                                                                 isEmitting(_isEmitting) {
                Vec3 min(fminf(a.x, fminf(b.x, c.x)),
                         fminf(a.y, fminf(b.y, c.y)),
                         fminf(a.z, fminf(b.z, c.z)));
                Vec3 max(fmaxf(a.x, fmaxf(b.x, c.x)),
                         fmaxf(a.y, fmaxf(b.y, c.y)),
                         fmaxf(a.z, fmaxf(b.z, c.z)));

                midPoint = (a + b + c) / 3;
                boundingBox = BoundingBox(min, max);

                calculateNormal();
            }

            __device__ __host__ Triangle(Vec3 _a, Vec3 _b, Vec3 _c, Vec3 _color) : Triangle(_a, _b, _c, _color, false) {}

            __device__ __host__ void calculateNormal() {
                ab = b - a;
                ac = c - a;

                normal = ab.cross(ac);
                normal.normalize();
            }

            // Moeller-Trumbore algorithm
            __device__ __host__ bool intersect(const Ray &ray, float &distance) const {
                Vec3 pvec = ray.direction.cross(ac);
                float det = ab.dot(pvec);

                if (det < EPSILON)
                    return false;

                float invDet = 1 / det;
                Vec3 tvec = ray.position - a;
                float u = tvec.dot(pvec) * invDet;
                if (u < 0.f || u > 1.f)
                    return false;

                Vec3 qvec = tvec.cross(ab);
                float v = ray.direction.dot(qvec) * invDet;
                if (v < 0.f || u + v > 1.f)
                    return false;

                distance = ac.dot(qvec) * invDet;
                return true;
            }
        };
    }

    __device__ inline int getGlobalIdx() {
        int blockID = blockIdx.y * gridDim.x + blockIdx.x;
        return blockID * blockDim.x + threadIdx.x;
    }

    __device__ bool intersectKDTreeRecursive(KDTreeGPUNode* nodes, size_t currentIdx, cuda::Triangle* triangles, size_t* triangleIndices, cuda::Ray ray, size_t &intersectedTriIdx, float &distance) {
//        bool hit = false;
//        size_t minTriangleIdx = 0;
//
//        KDTreeGPUNode current = nodes[0];
//        cuda::BoundingBox* bb = reinterpret_cast<cuda::BoundingBox*>(&current.bb);
//
//        float lambdaEntry, lambdaExit;
//        if (!bb->intersect(ray, lambdaEntry, lambdaExit)) {
//            return false;
//        }
//
//        while(lambdaEntry < lambdaExit) {
//            cuda::Vec3 pEntry = ray.position + ray.direction * lambdaEntry;
//
//            while (!current.isLeaf) {
//                if (pEntry[current.splitPlane.axis] < current.splitPlane.distance) {
//                    current = nodes[current.leftChildIndex];
//                } else {
//                    current = nodes[current.rightChildIndex];
//                }
//            }
//
//            size_t curIdxIdx = current.firstTriangleIdxIdx;
//            for(int i = 0; i < current.numTriangles; i++, curIdxIdx++) {
//                cuda::Triangle tri = triangles[triangleIndices[curIdxIdx]];
//                float tmpDist;
//                if(tri.intersect(ray, tmpDist)) {
//                    if(tmpDist <= lambdaExit && tmpDist >= lambdaEntry) {
//                        hit = true;
//                        lambdaExit = tmpDist;
//                        minTriangleIdx = triangleIndices[curIdxIdx];
//                    }
//                }
//            }
//
//            // Exit the leaf
//            float tmpEntry; // Not needed
//            bb = reinterpret_cast<cuda::BoundingBox*>(&current.bb);
//
//            bb->intersect(ray, tmpEntry, lambdaEntry);
//            cuda::Vec3 outPoint = ray.position + ray.direction * lambdaEntry;
//
//            float minDist = cuda::INFTY;
//            size_t minDistFace = 0;
//            for(size_t i = 0; i < 3; i++)
//            {
//                float tmpDist = outPoint[i] - bb->min[i];
//                if (tmpDist < minDist)
//                {
//                    minDist = tmpDist;
//                    minDistFace = i * 2;
//                }
//
//                tmpDist = bb->max[i] - outPoint[i];
//                if (tmpDist < minDist)
//                {
//                    minDist = tmpDist;
//                    minDistFace = (i * 2) + 1;
//                }
//            }
//
//            if (current.ropes[minDistFace] == -1)
//            {
//                break;
//            }
//            current = nodes[current.ropes[minDistFace]];
//        }
//
//        if (hit) {
//            intersectedTriIdx = minTriangleIdx;
//            distance = lambdaExit;
//            return true;
//        }
//        return false;

        KDTreeGPUNode current = nodes[currentIdx];
        cuda::BoundingBox* bb = (cuda::BoundingBox*)&current.bb;

        if(!bb->intersect(ray)) {
            return false;
        }

        if(!current.isLeaf) {
            bool hitLeft = false;
            bool hitRight = false;
            if(current.leftChildIndex != -1) {
                hitLeft = intersectKDTreeRecursive(nodes, current.leftChildIndex, triangles, triangleIndices, ray, intersectedTriIdx, distance);
            }

            if(current.rightChildIndex != -1) {
                hitRight = intersectKDTreeRecursive(nodes, current.rightChildIndex, triangles, triangleIndices, ray, intersectedTriIdx, distance);
            }
            return hitLeft || hitRight;
        }

        bool hit = false;
        float minDist = cuda::INFTY;
        size_t minTriangleIdx = 0;

        for(int i = 0; i < current.numTriangles; i++) {
            cuda::Triangle tri = triangles[triangleIndices[current.firstTriangleIdxIdx + i]];
            float tmpDist;
            if(tri.intersect(ray, tmpDist)) {
                if(tmpDist < minDist && tmpDist > 0) {
                    hit = true;
                    minDist = tmpDist;
                    minTriangleIdx = triangleIndices[current.firstTriangleIdxIdx + i];
                }
            }
        }

        if (hit && minDist < distance) {
            intersectedTriIdx = minTriangleIdx;
            distance = minDist;
            return true;
        }
        return false;
    }

    __device__ cuda::Vec3 evaluatePathRecursiveDiffuse(KDTreeGPUNode* nodes, cuda::Triangle* triangles, size_t* triangleIndices, cuda::Vec3* backgroundColor, curandState &s, size_t maxDepth, size_t numSamples, cuda::Ray &ray, size_t depth) {
        size_t intersectedTriangleIdx = 0;
        float distance = cuda::INFTY;
        if (!intersectKDTreeRecursive(nodes, 0, triangles, triangleIndices, ray, intersectedTriangleIdx, distance)) {
            return *backgroundColor;
        }

        cuda::Triangle intersectedTriangle = triangles[intersectedTriangleIdx];

        if (intersectedTriangle.isEmitting) {
            return intersectedTriangle.color;
        } else if (depth >= maxDepth) {
            return cuda::Vec3(0.f, 0.f, 0.f);
        }

        cuda::Vec3 intersectionPos = ray.position + ray.direction * distance;
        cuda::Vec3 color(0.f, 0.f, 0.f);

        for (size_t i = 0; i < numSamples; i++) {
            cuda::Ray newRay(intersectionPos, sampleHemisphere(intersectedTriangle.normal, curand_uniform(&s), curand_uniform(&s)));
            color = color + evaluatePathRecursiveDiffuse(nodes, triangles, triangleIndices, backgroundColor, s, maxDepth, numSamples, newRay, depth + 1);
        }

        return ((color * intersectedTriangle.color) / numSamples);
    }

    __device__ cuda::Vec3 evaluatePathRecursive(KDTreeGPUNode* nodes, cuda::Triangle* triangles, size_t* triangleIndices, cuda::Vec3* backgroundColor, curandState &s, size_t maxDepth, size_t numSamples, cuda::Ray &ray, size_t depth) {
        size_t intersectedTriangleIdx = 0;
        float distance = cuda::INFTY;
        if (!intersectKDTreeRecursive(nodes, 0, triangles, triangleIndices, ray, intersectedTriangleIdx, distance)) {
            return *backgroundColor;
        }

        cuda::Triangle intersectedTriangle = triangles[intersectedTriangleIdx];

        if (intersectedTriangle.isEmitting) {
            return intersectedTriangle.color;
        } else if (depth >= maxDepth) {
            return cuda::Vec3(0.f, 0.f, 0.f);
        }

        cuda::Vec3 intersectionPos = ray.position + ray.direction * distance;

        cuda::Ray newRay(intersectionPos, sampleHemisphere(intersectedTriangle.normal, curand_uniform(&s), curand_uniform(&s)));
        cuda::Vec3 color = evaluatePathRecursive(nodes, triangles, triangleIndices, backgroundColor, s, maxDepth, numSamples, newRay, depth + 1);

        return color * intersectedTriangle.color;
    }

    __global__ void evaluatePathKernelDiffuse(KDTreeGPUNode *nodes, cuda::Triangle *triangles, size_t *triangleIndices,
                                       cuda::Ray *rays, cuda::Vec3 *backgroundColor, size_t maxDepth, size_t numSamples,
                                       size_t start, size_t end, unsigned char *image) {
        int pixel = start + getGlobalIdx();

        if (pixel >= end) {
            return;
        }

        curandState s;
        curand_init(pixel, pixel, 0, &s);

        cuda::Vec3 color = evaluatePathRecursiveDiffuse(nodes, triangles, triangleIndices, backgroundColor, s, maxDepth, numSamples, rays[pixel], 0).clamp(0.f, 1.f);

        color = (color * 255.f) + 0.5f;

        image[pixel * 3] = color[0];
        image[pixel * 3 + 1] = color[1];
        image[pixel * 3 + 2] = color[2];
    }

    __global__ void evaluatePathKernel(KDTreeGPUNode *nodes, cuda::Triangle *triangles, size_t *triangleIndices,
                                       cuda::Ray *rays, cuda::Vec3 *backgroundColor, size_t maxDepth, size_t numSamples,
                                       size_t start, size_t end, unsigned char *image) {
        int pixel = start + getGlobalIdx();

        if (pixel >= end) {
            return;
        }

        curandState s;
        curand_init(pixel, pixel, 0, &s);

        cuda::Vec3 color(0.f, 0.f, 0.f);

        for(size_t i = 0; i < numSamples; i++)
        {
            color += evaluatePathRecursive(nodes, triangles, triangleIndices, backgroundColor, s, maxDepth, numSamples, rays[pixel], 0);
        }

        color = color / numSamples;
        color = (color.clamp(0.f, 1.f) * 255.f) + 0.5f;

        image[pixel * 3] = color[0];
        image[pixel * 3 + 1] = color[1];
        image[pixel * 3 + 2] = color[2];
    }

    dim3 getNeededGridDimension(int neededSize) {

        dim3 maxGridDimension(65535, 65535, 65535);

        dim3 dimGrid;

        if (neededSize > maxGridDimension.x) {
            dimGrid = dim3(maxGridDimension.x, (neededSize + maxGridDimension.x - 1) / maxGridDimension.x);
        } else {
            dimGrid = dim3(neededSize);
        }
        return dimGrid;
    }

    void copyKDTree(KDTreeGPU &kdTree, KDTreeGPUNode** d_kdNodes, cuda::Triangle** d_triangles, size_t** d_triangleIndices) {

        size_t nodeSize = kdTree.nodes.size() * sizeof(KDTreeGPUNode);
        CUDA_ERR_CHECK(cudaMalloc((void **)d_kdNodes, nodeSize));
        size_t triSize = kdTree.triangles.size() * sizeof(Triangle);
        CUDA_ERR_CHECK(cudaMalloc((void **)d_triangles, triSize));
        size_t indexSize = kdTree.triangleIndices.size() * sizeof(size_t);
        CUDA_ERR_CHECK(cudaMalloc((void **)d_triangleIndices, indexSize));

        CUDA_ERR_CHECK(cudaMemcpy(*d_kdNodes, kdTree.nodes.data(), nodeSize, cudaMemcpyHostToDevice));
        CUDA_ERR_CHECK(cudaMemcpy(*d_triangles, kdTree.triangles.data(), triSize, cudaMemcpyHostToDevice));
        CUDA_ERR_CHECK(cudaMemcpy(*d_triangleIndices, kdTree.triangleIndices.data(), indexSize, cudaMemcpyHostToDevice));
    }

    void pathTraceOnGPURecursive(Image &image, const Scene &scene) {
        if (scene.primRays.empty()) {
            return;
        }

        const std::vector<Ray> &rays = scene.primRays;

        size_t limit;
        cudaDeviceGetLimit(&limit, cudaLimitStackSize);
        limit = 20 * 1024;
        CUDA_ERR_CHECK( cudaDeviceSetLimit(cudaLimitStackSize, limit) );

        KDTreeGPU gpuTree = buildGPUTreeFromKDTree(scene.triangleDB);

        KDTreeGPUNode* d_kdNodes = nullptr;
        cuda::Triangle* d_triangles = nullptr;
        size_t* d_triangleIndices = nullptr;
        copyKDTree(gpuTree, &d_kdNodes, &d_triangles, &d_triangleIndices);

        cuda::Ray* d_rays = nullptr;
        unsigned char* d_image = nullptr;
        cuda::Vec3* d_backgroundColor = nullptr;

        CUDA_ERR_CHECK( cudaMalloc((void **) &d_rays, rays.size() * sizeof(Ray)) );
        CUDA_ERR_CHECK( cudaMemcpy(d_rays, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice) );

        CUDA_ERR_CHECK( cudaMalloc((void **) &d_image, rays.size() * sizeof(unsigned char) * 3) );

        CUDA_ERR_CHECK( cudaMalloc((void **) &d_backgroundColor, sizeof(Vec3)) );
        CUDA_ERR_CHECK( cudaMemcpy(d_backgroundColor, &scene.backgroundColor, sizeof(Vec3), cudaMemcpyHostToDevice) );

        dim3 dimBlock(256);
        size_t batchSize = 1024;
        size_t imageSize = scene.width * scene.height;
        for(size_t batchCtr = 0; batchCtr * batchSize < imageSize; batchCtr++) {

            size_t start = batchCtr * batchSize;
            size_t end = std::min((batchCtr + 1) * batchSize, imageSize);
            size_t rayCount = end - start;

            dim3 dimGrid = getNeededGridDimension((int)std::ceil((float)rayCount / (float)dimBlock.x));

            if (scene.ptMode == PtMode::DIFFUSE) {
                evaluatePathKernelDiffuse<<<dimGrid, dimBlock>>>(d_kdNodes, d_triangles, d_triangleIndices, d_rays, d_backgroundColor, scene.maxDepth, scene.numSamples, start, end, d_image);
            } else {
                evaluatePathKernel<<<dimGrid, dimBlock>>>(d_kdNodes, d_triangles, d_triangleIndices, d_rays, d_backgroundColor, scene.maxDepth, scene.numSamples, start, end, d_image);
            }
        }

        CUDA_ERR_CHECK( cudaPeekAtLastError() );
        CUDA_ERR_CHECK( cudaFree(d_rays) );
        CUDA_ERR_CHECK( cudaFree(d_backgroundColor) );

        image.resize(rays.size() * 3);
        CUDA_ERR_CHECK( cudaMemcpy(image.data(), d_image, rays.size() * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost) );
        CUDA_ERR_CHECK( cudaFree(d_image) );

        CUDA_ERR_CHECK( cudaFree(d_kdNodes) );
        CUDA_ERR_CHECK( cudaFree(d_triangles) );
        CUDA_ERR_CHECK( cudaFree(d_triangleIndices) );

    }

    bool isComputableOnGPU(const Scene &scene) {
        const std::vector<Ray> &rays = scene.primRays;
        size_t neededMem = 0;
        size_t mem_free, mem_tot;
        bool isComputableOnGPU = false;

        neededMem += rays.size() * sizeof(Ray); // rays on GPU
        neededMem += rays.size() * sizeof(unsigned char) * 3; // output image
        neededMem += sizeof(Vec3); // background color
        neededMem += scene.triangles.size() * sizeof(Triangle); // triangles
        size_t nodes = 2 * ((scene.triangles.size() + 5) / 6) - 1;
        neededMem += nodes * sizeof(KDTreeGPUNode); // GPU KD tree

        cudaMemGetInfo  (&mem_free, & mem_tot);
        isComputableOnGPU = mem_free >= neededMem;

        std::cout << "\tEstimated needed memory on GPU: " << neededMem / (double)1000000000 << "GB\n";
        std::cout << "\tFree memory on GPU: " << mem_free / (double)1000000000 << "GB\n";
        std::cout << "\tRunning on GPU possible: ";
        if(isComputableOnGPU) {
            std::cout << "yes";
        } else {
            std::cout << "no";
        }
        std::cout << std::endl;
        return isComputableOnGPU;
    }

}
