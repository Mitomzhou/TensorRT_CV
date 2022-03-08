# TensorRT -  CV
TensorRT_CV是使用TensorRT的高性能计算机视觉推理框架，提供跨平台C++接口,是tensorRT_Pro的Tiny版。详情请查看原版：[[tensorRT_Pro]](https://github.com/shouxieai/tensorRT_Pro/)
### 一、环境安装

- [x] Python3.8 
- [x] TensorRT-8.0.1.6
- [x] cudnn-8.2.2.26
- [x] cuda-10.2
- [x] pytorch-1.9.0
- [x] opencv-4.5.0
- [x] protobuf-3.11.4

1、protobuf-3.11.4安装
~~~bash
sudo apt-get install autoconf automake libtool curl
git clone https://github.com/google/protobuf.git --branch v3.11.4
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=/usr/local/protobuf-3.11.4
make -j8
make -j8 check
sudo make install
sudo ldconfig # refresh shared library cache.
# 配置环境变量
# protobuf-3.11.4
export PROTOBUF_HOME=/usr/local/protobuf-3.11.4
export PATH=$PATH:$PROTOBUF_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROTOBUF_HOME/lib
# 验证
protoc --version
~~~
2、cuda-10.2安装
~~~bash
安装最新的nvidia驱动
ubuntu-drivers devices
sudo apt-get install xxx
# 即便存在cuda-10.1也无妨,只需要来回切换/usr/local/cuda指向软链接和环境变量即可
sudo ./cuda_10.2.89_440.33.01_linux.run  
# 修改环境变量
# cuda-10.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export PATH=$PATH:/usr/local/cuda-10.2/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.2
# 测试
nvcc -V
~~~
3、cudnn-8.2.2.26安装
~~~bash
# 下载解压cudnn-10.2-linux-x64-v8.2.2.26.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-10.2/targets/x86_64-linux/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/targets/x86_64-linux/lib
sudo chmod a+x /usr/local/cuda-10.2/targets/x86_64-linux/include/cudnn*.h
sudo chmod a+x /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn*
~~~
4、pytorch安装
~~~bash
# CUDA 10.2
sudo pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# 离线
https://download.pytorch.org/whl/cu102/torch-1.9.0%2Bcu102-cp38-cp38-linux_x86_64.whl
https://download.pytorch.org/whl/cu102/torchvision-0.10.0%2Bcu102-cp38-cp38-linux_x86_64.whl
~~~
5、opencv-4.5.0安装
~~~bash
下载opencv-4.5.0.zip (https://opencv.org/releases/ )  
cd opencv-4.5.0     mkdir build    cd build
cmake .. -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.5.0
make -j8
sudo make install

https://blog.csdn.net/PecoHe/article/details/97476135
cd /usr/local/lib
sudo mkdir pkgconfig
cd pkgconfig
sudo touch opencv.pc
sudo vi /usr/local/lib/pkgconfig/opencv.pc 添加如下内容
---------------------------------------------------
prefix=/usr/local/opencv-4.5.0
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib
Name: opencv
Description: The opencv library
Version:4.5.0
Cflags: -I${includedir}/opencv4
Libs: -L${libdir} -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann  -lopencv_core
----------------------------------------------------
设置环境变量 sudo vi ~/.bashrc
# opencv-4.5.0
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
source ~/.bashrc
测试OpenCV
$ pkg-config opencv --modversion
4.5.0
~~~
6、安装TensorRT-8.0.1.6
~~~bash
下载并解压TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz
~~~

### 二、测试
1. 输入:yolov5s.onnx
2. 编译生成yolov5s.FP32.trtmodel模型文件
3. 推理获取结果

![test_yolov5](https://github.com/Mitomzhou/TensorRT_CV/blob/main/data/test_yolov5.png)

~~~cpp
#include <common/ilogger.hpp>
#include <infer/trt_infer.hpp>
#include <builder/trt_builder.hpp>
#include "app/yolo/yolo.hpp"

using namespace std;


void test_yolov5(){
    iLogger::set_log_level(iLogger::LogLevel::Verbose);

    int max_batch_size = 1;
    /** 模型编译，onnx到trtmodel(可离线) **/
    TRT::compile(
            TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
            max_batch_size,             /** 最大batch size        **/
            "yolov5s.onnx",             /** onnx文件，输入         **/
            "yolov5s.FP32.trtmodel"     /** trt模型文件，输出       **/
    );

    INFO("compile done!");

    /** 推理 **/
    auto yolo = Yolo::create_infer(
            "yolov5s.FP32.trtmodel",
            Yolo::Type::V5,
            0, 0.25f, 0.5f
            );
    auto image = cv::imread("car.jpg");
    auto bboxes = yolo->commit(image).get();

    for(auto& box : bboxes){
        uint8_t  r, g, b;
        tie(r, g, b) = iLogger::random_color(box.class_label);

        cv::rectangle(
                image,
                cv::Point(box.left, box.top),
                cv::Point(box.right, box.bottom),
                cv::Scalar(b, g, r),
                3
                );
    }
    cv::imwrite("car-yolov5.jpg", image);
}


int main()
{
    test_yolov5();
    return 0;
}
~~~