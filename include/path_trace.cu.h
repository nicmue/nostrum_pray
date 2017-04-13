#pragma once

#include "kd_tree_gpu.h"
#include "kd_tree_gpu_structs.h"
#include "geometry.h"
#include "pray.h"

namespace pray {
    /* DEPRECATED; CAUSES SEG FAULTS */
    // void pathTraceOnGPU(Image &image, const Scene &scene, std::vector<Ray> &rays);
    bool isComputableOnGPU(const Scene &scene);
    void pathTraceOnGPURecursive(Image &image, const Scene &scene);
}