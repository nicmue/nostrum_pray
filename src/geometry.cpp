#include "geometry.h"

namespace pray {
    Vec3 sampleHemisphere(const Vec3 &normal, float u1, float u2) {
        float theta = std::acos(std::sqrt(1 - u1));
        float phi = 2.f * PI * u2;

        Vec3 h = normal;

        if (std::abs(h.x) <= std::abs(h.y) && std::abs(h.x) <= std::abs(h.z)) {
            h.x = 1.f;
        } else if (std::abs(h.y) <= std::abs(h.x) && std::abs(h.y) <= std::abs(h.z)) {
            h.y = 1.f;
        } else {
            h.z = 1.f;
        }

        Vec3 x = h.cross(normal);
        x.normalize();

        Vec3 z = x.cross(normal);
        z.normalize();

        Vec3 s(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));

        Vec3 direction = x * s.x + normal * s.y + z * s.z;
        direction.normalize();
        return direction;
    }

    uint8_t isFaceParallelToAxis(Face &f, uint8_t axis) {
        switch(f) {
            case Face::left:
                if (axis == 0) {
                    return 1;
                }
            case Face::right:
                if (axis == 0) {
                    return 2;
                }

            case Face::front:
                if (axis == 1) {
                    return 1;
                }
            case Face::back:
                if (axis == 1) {
                    return 2;
                }

            case Face::top:
                if (axis == 2) {
                    return 1;
                }

            case Face::bottom:
                if (axis == 2) {
                    return 2;
                }
        }
        return 0;

    }
}