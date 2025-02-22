import cv2
import subprocess
import numpy as np

# RTSP 服务器地址
# rtsp_url = "rtsp://172.16.40.84:8554/1"

cap = cv2.VideoCapture('rtsp://172.16.40.84:553/live')
# cv::VideoCapture cap("rtsp://172.16.40.84:553/live", cv::CAP_FFMPEG);

# codec = cv2.VideoWriter_fourcc(*'H264')
# fps = 15
# frame_width = 640
# frame_height = 360
# out = cv2.VideoWriter(rtsp_url, codec, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # out.write(frame)
    # print("写入成功！！")
    cv2.imshow("1",frame)
    cv2.waitKey(1)
    

cap.release()
out.release()
