#include "scene_preprocessor.h"

#include "pray.h"
#include "kd_tree.h"
#include "kd_tree_sah.h"
#include "kd_tree_hybrid.h"
#include "bf_search.h"
#include "kd_tree_gpu.h"

#include <iostream>
#include <chrono>

#ifdef ENABLE_CUDA
#include "path_trace.cu.h"
#endif


namespace pray {
    void preprocessScene(Scene &scene) {
        std::cout << "\tInitialising triangles..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        initialiseTriangles(scene);
        auto end = std::chrono::steady_clock::now();
        std::cout << "\tDone: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        std::cout << "\tCalculating primary rays..." << std::endl;
        start = std::chrono::steady_clock::now();
        calculatePrimRays(scene);
        end = std::chrono::steady_clock::now();
        std::cout << "\tDone: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        if(scene.mode == Mode::PATHTRACING && scene.maxDepth > MAX_DEPTH_DIFFUSE)
        {
            std::cout << "\tUsing normal path tracing." << std::endl;
            scene.ptMode = PtMode::NORMAL;
        } else {
            std::cout << "\tUsing diffuse path tracing." << std::endl;
        }

#ifdef ENABLE_CUDA
        if (scene.mode == Mode::PATHTRACING && isComputableOnGPU(scene) &&
                scene.triangles.size() < MAX_NUM_TRIANGLES_GPU) {
            scene.useGPU = true;
        } else {
            scene.useGPU = false;
        }
#endif

        //scene.useGPU = true
        if (scene.triangles.size() < MIN_NUM_TRIANGLES_SAH || scene.triangles.size() >= MAX_NUM_TRIANGLES_SAH || scene.useGPU)
        {
            scene.triangleDB = std::make_unique<KDTree>();
        } else {
            scene.triangleDB = std::make_unique<KDTreeSAH>();
        }

        start = std::chrono::steady_clock::now();
        scene.triangleDB->build(std::move(scene.triangles));
        end = std::chrono::steady_clock::now();
        std::cout << "\t\t# nodes: " << scene.triangleDB->nnodes
                  << ", tree depth: " << scene.triangleDB->maxdepth
                  << ", leaf nodes: " << scene.triangleDB->leafNodes
                  << ", non-empty leaf nodes: " << scene.triangleDB->leafNodes - scene.triangleDB->emptyLeafNodes
                  << ", empty leaf nodes: " << scene.triangleDB->emptyLeafNodes
                  << ", nodes per leaf: " << scene.triangleDB->size() /
                                             static_cast<double>((scene.triangleDB->leafNodes - scene.triangleDB->emptyLeafNodes)) << std::endl;
        std::cout << "\tDone: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    }

    void initialiseTriangles(Scene &scene) {

        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < scene.triangles.size(); i++)
        {
            Triangle &curTri = scene.triangles[i];
            curTri.boundingBox.min = Vec3(std::min(curTri.a.x, std::min(curTri.b.x, curTri.c.x)),
                     std::min(curTri.a.y, std::min(curTri.b.y, curTri.c.y)),
                     std::min(curTri.a.z, std::min(curTri.b.z, curTri.c.z)));
            curTri.boundingBox.max = Vec3(std::max(curTri.a.x, std::max(curTri.b.x, curTri.c.x)),
                     std::max(curTri.a.y, std::max(curTri.b.y, curTri.c.y)),
                     std::max(curTri.a.z, std::max(curTri.b.z, curTri.c.z)));

            scene.triangles[i].midPoint = (curTri.a + curTri.b + curTri.c) / 3;

            // Calculate normal
            curTri.ab = curTri.b - curTri.a;
            curTri.ac = curTri.c - curTri.a;

            curTri.normal = curTri.ab.cross(curTri.ac);
            curTri.normal.normalize();
        }
    }

    void calculatePrimRays(Scene &scene) {
        scene.primRays.resize(scene.width * scene.height);

        Vec3 target = scene.camera_position + scene.camera_look;
        Vec3 up(0, 1, 0);
        Vec3 zaxis = target - scene.camera_position;
        zaxis.normalize();
        Vec3 xaxis = up.cross(zaxis);
        xaxis.normalize();
        Vec3 yaxis = zaxis.cross(xaxis);

        Matrix44 viewMatrix = Matrix44(
                xaxis.x, yaxis.x, zaxis.x, 0,
                xaxis.y, yaxis.y, zaxis.y, 0,
                xaxis.z, yaxis.z, zaxis.z, 0,
                -xaxis.dot(scene.camera_position), -yaxis.dot(scene.camera_position),
                -zaxis.dot(scene.camera_position), 1);

        Matrix44 inverseViewMatrix = viewMatrix.inverse();

        float invWidth = 1 / float(scene.width);
        float invHeight = 1 / float(scene.height);
        float aspectratio = scene.width / float(scene.height);
        float angle = std::tan(PI * 0.5f * scene.field_of_view / 180.f);

        // for each pixel of the image
        #pragma omp parallel for schedule(static) collapse(2)
        for (uint64_t yPixel = 0; yPixel < scene.height; yPixel++) {
            for (uint64_t xPixel = 0; xPixel < scene.width; xPixel++) {
                // compute primary ray direction
                float x = (2 * ((xPixel + 0.5f) * invWidth) - 1) * angle;
                float y = (1 - 2 * ((yPixel + 0.5f) * invHeight)) * angle / aspectratio;
                Vec3 directionCameraSpace(x, y, 1);
                Vec3 directionWorldSpace = inverseViewMatrix.multDirMatrix(directionCameraSpace);

                scene.primRays[xPixel + yPixel * scene.width] = Ray(scene.camera_position, directionWorldSpace);
            }
        }
    }
}
