set(SRC_DIR
        pray.cpp
        io.cpp
        kd_tree.cpp
        kd_tree_sah.cpp
        kd_tree_hybrid.cpp
        bf_search.cpp
        geometry_simd.cpp
        main.cpp
        geometry.cpp
        gpu_util.cpp
        scene_preprocessor.cpp)

set(INCLUDE_DIR
        ${Pray_SOURCE_DIR}/include
        ${Pray_SOURCE_DIR}/ext)

set(CUDA_SOURCES
        path_trace.cu)

include_directories(${INCLUDE_DIR})

if (WITH_CUDA)
    cuda_add_executable(Pray ${SRC_DIR} ${CUDA_SOURCES})
else()
    add_executable(Pray ${SRC_DIR})
endif()

#set_target_properties(Pray PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(Pray "-lstdc++fs")

if (WITH_OMP)
    target_compile_options(Pray PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(Pray ${OpenMP_CXX_FLAGS})
endif()

if (WITH_CUDA)
    target_compile_definitions(Pray PRIVATE WITH_CUDA)
endif()