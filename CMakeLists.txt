cmake_minimum_required(VERSION 3.17)
project(TensorRT_CV)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 如果要支持python则设置python路径 HAS_PYTHON: OFF/ON
set(HAS_PYTHON OFF)
set(PythonRoot "/usr/local/lib/python3.8/dist-packages/torch")
set(PythonName "python3.8")

# OpenCV
set(OpenCV_DIR   "/usr/local/opencv-4.5.0/")

# CUDA
# 不同显卡算力参考：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")
set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-10.2")
set(CUDNN_DIR    "/home/mitom/CUDALesson/cudnn-10.2-linux-x64-v8.2.2.26/cuda")

# TensorRT-8.0.1.6
set(TENSORRT_DIR "/home/mitom/CUDALesson/TensorRT-8.0.1.6")

# protobuf-3.11.4"
set(PROTOBUF_DIR "/usr/local/protobuf-3.11.4")


find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
        ${PROJECT_SOURCE_DIR}/tensorRT
        ${PROJECT_SOURCE_DIR}/tensorRT/common
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${PROTOBUF_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
)

link_directories(
        ${PROTOBUF_DIR}/lib
        ${TENSORRT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_DIR}/lib
)

# Python
# TODO

# set compile
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/tensorRT/*.cpp ${PROJECT_SOURCE_DIR}/app/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/tensorRT/*.cu ${PROJECT_SOURCE_DIR}/app/*.cu)

cuda_add_library(plugin_list SHARED ${cuda_srcs} ${cpp_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})


add_executable(TensorRT_CV main.cpp)

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(TensorRT_CV nvinfer nvinfer_plugin)
target_link_libraries(TensorRT_CV cuda cublas cudart cudnn)
target_link_libraries(TensorRT_CV protobuf pthread plugin_list)
target_link_libraries(TensorRT_CV ${OpenCV_LIBS})

