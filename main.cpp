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