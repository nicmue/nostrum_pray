#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>
#include <map>

using size_t = std::size_t;
using uint_8 = std::uint8_t;

#include "kd_tree_sah.h"

namespace pray {
    class KDTreeHybrid : public KDTreeSAH {
    public:
        virtual void build(const std::vector<Triangle> &triangles) override;
        virtual void build(std::vector<Triangle> &&triangles) override;

    private:
        float traversalCost = 1.f;
        float triangleIntersectionCost = 1.5f;


        void build();

        std::unique_ptr<KDTreeElement> buildNoSAHRecursive(std::vector<size_t> &triIndices, size_t depth);
    };
}
