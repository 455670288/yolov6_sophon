//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "yolov6.hpp"
using json = nlohmann::json;
using namespace std;

// #define DEBUG 1

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include <vector>
#include <unordered_set>
#include <nadjieb/mjpeg_streamer.hpp>


const float person_thres = 0.5;               //0
const float head_helmet_thres =0.5;           //1
const float head_thres = 0.5;                 //2
const float reflective_clothes_thres = 0.5;   //3
const float smoking_thres = 0.5;              //4
const float calling_thres = 0.5;              //5
const float falling_thres = 0.5;              //6
const float face_mask_thres = 0.5;            //7
const float car_thres = 0.5;                  //8
const float bicycle_thres = 0.5;              //9
const float motorcycle_thres = 0.5;           //10
const float fumes_thres = 0.5;                //11
const float fire_thres = 0.5;                 //12
const float head_hat_thres = 0.5;             //13
const float normal_clothes_thres = 0.5;       //14
const float face_thres = 0.5;                 //15
const float play_phone_thres = 0.5;           //16
const float other_thres = 0.5;                //17
const float knife_thres = 0.5;                //18


// 每个类别的筛选阈值
std::vector<float> conf_thresholds {person_thres, head_helmet_thres, head_thres, reflective_clothes_thres, smoking_thres,
calling_thres, falling_thres, face_mask_thres, car_thres, bicycle_thres, motorcycle_thres, fumes_thres, fire_thres, head_hat_thres, normal_clothes_thres,face_thres,play_phone_thres,other_thres,
    knife_thres};

std::unordered_set<int> filtered_classes;

int main(int argc, char *argv[]){

    /*
    模型推理
    */
    cout.setf(ios::fixed);
    // get params
    const char *keys="{bmodel | ../../models/BM1684X/yolov11s_fp32_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{conf_thresh | 0.25 | confidence threshold for filter boxes}"
    "{nms_thresh | 0.7 | iou threshold for nms}"
    "{help | 0 | print help information.}"
    "{input | ../../datasets/test | input path, images direction or video file path}"
    "{classnames | ../../datasets/coco.names | class names file path}"
    "{classes_filter | 0 | filter or not}"
    "{classes_filter_list | 0 | list of classes_filter}";
    cv::CommandLineParser parser(argc, argv, keys);
    
    bool cls_filter = parser.get<bool>("classes_filter");
    if(cls_filter) {
        //输入string进行传递
        std::string numbers_str = parser.get<std::string>("classes_filter_list");
        if(!numbers_str.empty()){
            std::stringstream ss(numbers_str);
            int num;
            while (ss >> num){
                filtered_classes.insert(num);
            }
        }
    }





    if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
    }
    string bmodel_file = parser.get<string>("bmodel");
    string input = parser.get<string>("input");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
    }
    string coco_names = parser.get<string>("classnames");
    if (stat(coco_names.c_str(), &info) != 0) {
    cout << "Cannot find classnames file." << endl;
    exit(1);
    }
    if (stat(input.c_str(), &info) != 0){
    cout << "Cannot find input path." << endl;
    // exit(1);
    }

    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    cout << "set device id: "  << dev_id << endl;
    bm_handle_t h = handle->handle();

    // load bmodel
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

    // initialize net
    YoLoV6 yolov6(bm_ctx);
    CV_Assert(0 == yolov6.Init(
        parser.get<float>("conf_thresh"),
        parser.get<float>("nms_thresh"),
        coco_names,conf_thresholds));

    // profiling
    TimeStamp yolov6_ts;
    TimeStamp *ts = &yolov6_ts;
    yolov6.enableProfile(&yolov6_ts);

    // get batch_size
    int batch_size = yolov6.batch_size();

    // creat save path
    if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
    mkdir("results/images", S_IRWXU);
    
    VideoDecFFM decoder;
    decoder.openDec(&h, input.c_str());
    int id = 0;
    vector<bm_image> batch_imgs;
    vector<YoLoV6BoxVec> boxes;
    
    nadjieb::MJPEGStreamer streamer;
    streamer.start(8010);
    while(true){
            bm_image *img = decoder.grab();
            if (!img)
                break;
            batch_imgs.push_back(*img);
            delete img;
            img = nullptr;
            if ((int)batch_imgs.size() == batch_size){
                // predict
                CV_Assert(0 == yolov6.Detect(batch_imgs, boxes));
                for(int i = 0; i < batch_size; i++){
                    id++;
                    cout << id << ", det_nums: " << boxes[i].size() << endl;
                    if (batch_imgs[i].image_format != 0){
                        bm_image frame;
                        bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
                        bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
                        bm_image_destroy(batch_imgs[i]);
                        batch_imgs[i] = frame;
                    }
                    for (auto bbox : boxes[i]) {
                        int bboxwidth = bbox.x2-bbox.x1;
                        int bboxheight = bbox.y2-bbox.y1;
                        // draw image
                        if(bbox.score > 0.25 && cls_filter){
                            if(filtered_classes.find(bbox.class_id) != filtered_classes.end()){
                              yolov6.draw_bmcv(h, bbox.class_id, bbox.score, bbox.x1, bbox.y1, bboxwidth, bboxheight, batch_imgs[i]);
                            }
                        }
                        else if (bbox.score > 0.25){
                            yolov6.draw_bmcv(h, bbox.class_id, bbox.score, bbox.x1, bbox.y1, bboxwidth, bboxheight, batch_imgs[i]);
                        }
                            
                    }
                    
                    // 显示视频流
                    void* jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret =  bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                    std::vector<uchar> jpeg_buffer(static_cast<uchar*>(jpeg_data), static_cast<uchar*>(jpeg_data) + out_size);

                    //http://x.x.x.x:8010/bgr 显示
                    streamer.publish("/bgr", std::string(jpeg_buffer.begin(), jpeg_buffer.end()));
                    // cv::Mat img = cv::imdecode(jpeg_buffer, cv::IMREAD_COLOR);
                    free(jpeg_data);

                    // 转换为字节流写入到文件
                    // string img_file = "results/images/" + to_string(id) + ".jpg";
                    // void* jpeg_data = NULL;
                    // size_t out_size = 0;
                    // int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                    // if (ret == BM_SUCCESS) {
                    //     FILE *fp = fopen(img_file.c_str(), "wb");
                    //     fwrite(jpeg_data, out_size, 1, fp);
                    //     fclose(fp);
                    // }
                    // free(jpeg_data);

                    bm_image_destroy(batch_imgs[i]);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                // print speed
                time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
                yolov6_ts.calbr_basetime(base_time);
                yolov6_ts.build_timeline("yolov6 test");
                yolov6_ts.show_summary("yolov6 test");
                yolov6_ts.clear();

                batch_imgs.clear();
                boxes.clear();
            }
    }
        streamer.stop();

   
    // print speed
    // time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    // yolov6_ts.calbr_basetime(base_time);
    // yolov6_ts.build_timeline("yolov6 test");
    // yolov6_ts.show_summary("yolov6 test");
    // yolov6_ts.clear();
    return 0;
}
