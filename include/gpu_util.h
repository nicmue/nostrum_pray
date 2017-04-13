#pragma once

#include "kd_tree.h"
#include "kd_tree_gpu.h"
#include <memory>
#include <vector>

namespace pray {

    KDTreeGPU buildGPUTreeFromKDTree(const std::unique_ptr<SpatialDatabase> &kdtree);
    void gpuCreateRopes(KDTreeGPU &kdTree, int curNodeIdx, std::array<int, 6> currentRopes);
    void gpuOptimizeRope(KDTreeGPU &kdTree, int &ropeIdx, Face f, BoundingBox &bb);

    struct ColorTreeNode {
        Vec3 color;
        std::vector<ColorTreeNode> childs;

        ColorTreeNode() : color(Vec3(0.f, 0.f, 0.f)), childs(std::vector<ColorTreeNode>()) {}
    };

    Vec3 colorDFS(const ColorTreeNode &node);
}