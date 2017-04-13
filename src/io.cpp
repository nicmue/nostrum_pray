#define TINYOBJLOADER_IMPLEMENTATION

#include "io.h"

/* Dependency: JSON for Modern C++
*  https://nlohmann.github.io/json/
*/
#include "json.hpp"

using json = nlohmann::json;

/* Dependency: Tiny but powerful single file wavefront obj loader
*  https://github.com/syoyo/tinyobjloader
*/
#include "tiny_obj_loader.h"

#include <cstdint>

using std::uint64_t;
using std::uint8_t;

#include <tiny_obj_loader.h>

namespace fs = std::experimental::filesystem;

namespace pray {
    void readSceneDescription(const fs::path &inputFileName, Scene &scene) {
        auto basePath = inputFileName.parent_path();

        std::ifstream fin(inputFileName);
        if (!fin.good()) {
            std::cerr << "Error opening file: " << inputFileName << "\n";
            exit(EXIT_FAILURE);
        }
        json json_input = json::parse(fin);
        fin.close();

        std::string error;
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        // get full path to the obj file
        auto objFileName = basePath / json_input["obj_file"].get<std::string>();
        // base path of the mtl files
        auto mtlDir = basePath.string() + fs::path::preferred_separator;

        bool obj_succ = tinyobj::LoadObj(
                &attrib, &shapes, &materials, &error, objFileName.c_str(),
                (basePath.empty() ? nullptr : mtlDir.c_str()));

        // Warnings might be present even on success
        if (!error.empty())
            std::cerr << error << "\n";
        if (!obj_succ) {
            std::cerr << "Error loading file: " << json_input["obj_file"] << "\n";
            exit(EXIT_FAILURE);
        }

        // JSON input

        if (json_input["method"] == "path_tracing") {
            scene.mode = Mode::PATHTRACING;
            scene.maxDepth = json_input["max_depth"];
            scene.numSamples = json_input["num_samples"];
        } else {
            scene.mode = Mode::RAYTRACING;
            scene.maxDepth = 0;
            scene.numSamples = 0;
        }

        // Image resolution
        scene.width = json_input["resolution_x"];
        scene.height = json_input["resolution_y"];

        // Field of view is the angle between the rays for the rightmost and leftmost
        // pixels. The vertical fov is determined from the resolution ratio.
        scene.field_of_view = json_input["fov"];

        // Position of the camera in the scene
        scene.camera_position = Vec3(json_input["camera_position"][0],
                                    json_input["camera_position"][1],
                                    json_input["camera_position"][2]);

        // Orientation of the camera in the scene. The y-axis of the camera view is
        // always the "up"-axis wrt. to the image.
        scene.camera_look = Vec3(json_input["camera_look"][0], json_input["camera_look"][1],
                                json_input["camera_look"][2]);

        // Load point lights
        if (!scene.mode == Mode::PATHTRACING) {
            for (auto &l : json_input["lights"]) {
                Vec3 position(l["position"][0], l["position"][1], l["position"][2]);
                Vec3 color(l["color"][0], l["color"][1], l["color"][2]);
                scene.lights.push_back(LightSource(position, color));
            }
        }
        // Background color of the scene. Colors are given as floating point RGB
        // values in the interval [0.0, 1.0]
        scene.backgroundColor = Vec3(json_input["background"][0], json_input["background"][1],
                                    json_input["background"][2]);

        // Loop over shapes
        scene.triangles.reserve(shapes.size()); // TODO: better init size?
        for (size_t i = 0; i < shapes.size(); i++) {
            // Loop over faces(triangles)
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
                auto fv = shapes[i].mesh.num_face_vertices[f];
                assert(fv == 3);

                // Loop over vertices in the face. (unrolled)
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + 0];
                Vec3 v0(attrib.vertices[3 * idx.vertex_index + 0],
                       attrib.vertices[3 * idx.vertex_index + 1],
                       attrib.vertices[3 * idx.vertex_index + 2]);
                idx = shapes[i].mesh.indices[index_offset + 1];
                Vec3 v1(attrib.vertices[3 * idx.vertex_index + 0],
                       attrib.vertices[3 * idx.vertex_index + 1],
                       attrib.vertices[3 * idx.vertex_index + 2]);
                idx = shapes[i].mesh.indices[index_offset + 2];
                Vec3 v2(attrib.vertices[3 * idx.vertex_index + 0],
                       attrib.vertices[3 * idx.vertex_index + 1],
                       attrib.vertices[3 * idx.vertex_index + 2]);
                index_offset += fv;

                // Per-face material, color color.
                Vec3 emission = materials[shapes[i].mesh.material_ids[f]].emission;
                if (emission == Vec3(0.f, 0.f, 0.f)) {
                    Vec3 diffuse = materials[shapes[i].mesh.material_ids[f]].diffuse;
                    scene.triangles.push_back(Triangle(v0, v1, v2, diffuse));
                } else {
                    scene.triangles.push_back(Triangle(v0 , v1, v2, emission, true));
                }
            }
        }
    }
}