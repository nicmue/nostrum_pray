#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <array>

using std::uint8_t;

namespace pray {
    static constexpr float EPSILON = 1.e-4f;
    static const float PI = static_cast<float>(std::acos(-1));

    inline float clamp(float lo, float hi, const float &v) { return std::max(lo, std::min(hi, v)); }

    struct Vec3 {
        float x;
        float y;
        float z;

        Vec3() = default;

        Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

        Vec3(const float v[3]) : x(v[0]), y(v[1]), z(v[2]) {}

        void normalize() {
            float nor2 = norm();
            if (nor2 > 0) {
                float invNor = 1 / std::sqrt(nor2);
                x *= invNor, y *= invNor, z *= invNor;
            }
        }

        float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

        Vec3 cross(const Vec3 &v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

        Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
        Vec3 operator+=(const Vec3 &v) { x += v.x, y += v.y, z += v.z; return *this;}

        Vec3 operator+(const float &r) const { return Vec3(x + r, y + r, z + r); }

        Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }

        Vec3 operator-() const { return Vec3(-x, -y, -z); }

        Vec3 operator*(const float &r) const { return Vec3(x * r, y * r, z * r); }

        Vec3 operator*(const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

        Vec3 operator/(const float &r) const { return Vec3(x / r, y / r, z / r); }

        bool operator==(const Vec3 &v) const { return v.x == x && v.y == y && v.z == z; }

        const float &operator[](int i) const {
            assert(i >= 0 && i < 3);
            return (&x)[i];
        }

        float norm() const { return x * x + y * y + z * z; }

        float length() const { return std::sqrt(norm()); }

        Vec3 clamp(float lo, float hi) {
            return Vec3(pray::clamp(lo, hi, x), pray::clamp(lo, hi, y), pray::clamp(lo, hi, z));
        }

        Vec3 inverse() {
            return Vec3(1 / x, 1 / y, 1 / z);
        }
    };


    struct Ray {

        Vec3 position;
        Vec3 direction;
        Vec3 inverseDirection;

        Ray() = default;

        Ray(Vec3 _position, Vec3 _direction) : position(_position), direction(_direction) {
            direction.normalize();
            inverseDirection = direction.inverse();
        }
    };

    Vec3 sampleHemisphere(const Vec3 &normal, float u1, float u2);
    struct SplitPlane {
        uint8_t axis;	// splitting dimension
        float distance;	// splitting point

        SplitPlane(uint8_t _axis=255, float _distance=0) : axis(_axis), distance(_distance) {}

        bool operator==(const SplitPlane& sp) {
            return(axis == sp.axis && distance == sp.distance);
        }
    };

    struct BoundingBox {
        Vec3 min;
        Vec3 max;

        BoundingBox() = default;

        ~BoundingBox() = default;

        BoundingBox(Vec3 _min, Vec3 _max) : min(_min), max(_max) {}

        BoundingBox expand(const BoundingBox &bb) const {
            Vec3 tmpMin(std::min(min.x, bb.min.x), std::min(min.y, bb.min.y),
                        std::min(min.z, bb.min.z));
            Vec3 tmpMax(std::max(max.x, bb.max.x), std::max(max.y, bb.max.y),
                        std::max(max.z, bb.max.z));

            return BoundingBox(tmpMin, tmpMax);
        }

        BoundingBox clip(const BoundingBox &bb) const {
            Vec3 tmpMin(std::max(min.x, bb.min.x), std::max(min.y, bb.min.y), std::max(min.z, bb.min.z));
            Vec3 tmpMax(std::min(max.x, bb.max.x), std::min(max.y, bb.max.y), std::min(max.z, bb.max.z));

            auto b = BoundingBox(tmpMin, tmpMax);
            assert(bb.contains(b));

            return b;
        }

        bool isPlanar(uint8_t splitDim) {
            return min[splitDim] - max[splitDim] == 0;
        }

        uint8_t getLongestAxis() {
            float x = -1, y = -1, z = -1;

            x = max.x - min.x;
            y = max.y - min.y;
            z = max.z - min.z;

            if (x >= y && x >= z) {
                return 0;
            } else if (y >= x && y >= z) {
                return 1;
            } else {
                return 2;
            }
        }

        bool intersect(const Ray &ray) const {
            float tNear = std::numeric_limits<float>::lowest();
            float tFar = std::numeric_limits<float>::max();

            Vec3 t1 = (min - ray.position) * ray.inverseDirection;
            Vec3 t2 = (max - ray.position) * ray.inverseDirection;

            Vec3 tNear2(std::min(t1.x, t2.x), std::min(t1.y, t2.y), std::min(t1.z, t2.z));
            Vec3 tFar2(std::max(t1.x, t2.x), std::max(t1.y, t2.y), std::max(t1.z, t2.z));

            tNear = std::max(std::max(tNear2.x, tNear2.y), std::max(tNear2.z, tNear));
            tFar = std::min(std::min(tFar2.x, tFar2.y), std::min(tFar2.z, tFar));


            return tNear <= tFar;
        }

        bool intersect(const Ray &ray, float &tNear, float &tFar) const {
            tNear = std::numeric_limits<float>::lowest();
            tFar = std::numeric_limits<float>::max();

            Vec3 t1 = (min - ray.position) * ray.inverseDirection;
            Vec3 t2 = (max - ray.position) * ray.inverseDirection;

            Vec3 tNear2(std::min(t1.x, t2.x), std::min(t1.y, t2.y), std::min(t1.z, t2.z));
            Vec3 tFar2(std::max(t1.x, t2.x), std::max(t1.y, t2.y), std::max(t1.z, t2.z));

            tNear = std::max(std::max(tNear2.x, tNear2.y), std::max(tNear2.z, tNear));
            tFar = std::min(std::min(tFar2.x, tFar2.y), std::min(tFar2.z, tFar));

            return tNear <= tFar;
        }

        // surface area of a BoundinBox b
        float surfaceArea() const {
            float dx = max.x - min.x;
            float dy = max.y - min.y;
            float dz = max.z - min.z;

            return 2.f * (dx * dy + dx * dz + dy * dz);
        }

        // split a BoundingBox b in middle of a
        void split(const SplitPlane& p, BoundingBox& bl, BoundingBox& br) const {
            bl = BoundingBox(min, max);
            br = BoundingBox(min, max);

            if (p.axis == 0) {
                bl.max.x = p.distance;
                br.min.x = p.distance;
            } else if (p.axis == 1) {
                bl.max.y = p.distance;
                br.min.y = p.distance;
            } else {
                bl.max.z = p.distance;
                br.min.z = p.distance;
            }

            assert(contains(bl));
            assert(contains(br));
        }

        bool contains(const BoundingBox &b) const {
            return (min.x - EPSILON <= b.min.x && min.y - EPSILON <= b.min.y && min.z - EPSILON <= b.min.z)
                 && (max.x + EPSILON >= b.max.x && max.y + EPSILON >= b.max.y && max.z + EPSILON >= b.max.z);
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

        Triangle() = default;

        ~Triangle() = default;

        Triangle(Vec3 _a, Vec3 _b, Vec3 _c, Vec3 _color, bool _isEmitting) : a(_a), b(_b), c(_c), color(_color), isEmitting(_isEmitting) {
        }

        Triangle(Vec3 _a, Vec3 _b, Vec3 _c, Vec3 _color) : Triangle(_a, _b, _c, _color, false) {}

        // Moeller-Trumbore algorithm
        bool intersect(const Ray &ray, float &distance) const {
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

    enum Face {
        left,
        right,
        bottom,
        top,
        front,
        back
    };

    static constexpr std::array<Face, 6> allFaces{{Face::left, Face::right, Face::top, Face::bottom, Face::front, Face::back}};

    uint8_t isFaceParallelToAxis(Face &f, uint8_t axis);

    class Matrix44 {
    public:

        float x[4][4] = {{1, 0, 0, 0},
                         {0, 1, 0, 0},
                         {0, 0, 1, 0},
                         {0, 0, 0, 1}};

        Matrix44() {}

        Matrix44(float a, float b, float c, float d, float e, float f, float g, float h,
                 float i, float j, float k, float l, float m, float n, float o, float p) {
            x[0][0] = a;
            x[0][1] = b;
            x[0][2] = c;
            x[0][3] = d;
            x[1][0] = e;
            x[1][1] = f;
            x[1][2] = g;
            x[1][3] = h;
            x[2][0] = i;
            x[2][1] = j;
            x[2][2] = k;
            x[2][3] = l;
            x[3][0] = m;
            x[3][1] = n;
            x[3][2] = o;
            x[3][3] = p;
        }

        const float *operator[](uint8_t i) const { return x[i]; }

        float *operator[](uint8_t i) { return x[i]; }

        Vec3 multDirMatrix(const Vec3 &src) const {
            float a, b, c;

            a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0];
            b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1];
            c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2];

            return Vec3(a, b, c);
        }

        Matrix44 inverse() const {
            int i, j, k;
            Matrix44 s;
            Matrix44 t(*this);

            // Forward elimination
            for (i = 0; i < 3; i++) {
                int pivot = i;

                float pivotsize = t[i][i];

                if (pivotsize < 0)
                    pivotsize = -pivotsize;

                for (j = i + 1; j < 4; j++) {
                    float tmp = t[j][i];

                    if (tmp < 0)
                        tmp = -tmp;

                    if (tmp > pivotsize) {
                        pivot = j;
                        pivotsize = tmp;
                    }
                }

                if (pivotsize == 0) {
                    // Cannot invert singular matrix
                    return Matrix44();
                }

                if (pivot != i) {
                    for (j = 0; j < 4; j++) {
                        float tmp;

                        tmp = t[i][j];
                        t[i][j] = t[pivot][j];
                        t[pivot][j] = tmp;

                        tmp = s[i][j];
                        s[i][j] = s[pivot][j];
                        s[pivot][j] = tmp;
                    }
                }

                for (j = i + 1; j < 4; j++) {
                    float f = t[j][i] / t[i][i];

                    for (k = 0; k < 4; k++) {
                        t[j][k] -= f * t[i][k];
                        s[j][k] -= f * s[i][k];
                    }
                }
            }

            // Backward substitution
            for (i = 3; i >= 0; --i) {
                float f;

                if ((f = t[i][i]) == 0) {
                    // Cannot invert singular matrix
                    return Matrix44();
                }

                for (j = 0; j < 4; j++) {
                    t[i][j] /= f;
                    s[i][j] /= f;
                }

                for (j = 0; j < i; j++) {
                    f = t[j][i];

                    for (k = 0; k < 4; k++) {
                        t[j][k] -= f * t[i][k];
                        s[j][k] -= f * s[i][k];
                    }
                }
            }

            return s;
        }

        const Matrix44 &invert() {
            *this = inverse();
            return *this;
        }
    };
}
