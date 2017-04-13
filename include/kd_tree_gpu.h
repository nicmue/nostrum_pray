#pragma once

#include <memory>
#include <vector>

#include "kd_tree_gpu_structs.h"
#include "geometry.h"

namespace pray {

    class KDTreeGPU {
    public:
        std::vector<KDTreeGPUNode> nodes;
        std::vector<Triangle> triangles;

        std::vector<size_t> triangleIndices;

        KDTreeGPU() = default;
    };
}