cmake_minimum_required(VERSION 3.10)
project(yolov6_bmcv)

#自定义参数
set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/../../soc-sdk-1001)
set(TARGET_ARCH soc)

#可执行文件输出路径
# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# message("cur dir: ${PROJECT_SOURCE_DIR}")

#Usage: check_and_replace_path(SDK)
#Input src path, output absolute path.
function(check_and_replace_path ARG_NAME) 
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        message("Path ${PATH_TO_CHECK} exists")
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)
        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path!")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    # set(lib_DIR /usr/lib/x84_64-linux-gnu)
    # link_directories(${lib_DIR})
    # use ffmpeg
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    # use opencv
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories("../dependencies/include/")

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../dependencies/src/ SRC_FILES)
    add_executable(yolov6_bmcv.pcie ${SRC_FILES})
    target_link_libraries(yolov6_bmcv.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -std=c++11 -O2")
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(JPU_LIBS bmjpuapi bmjpulite)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("../dependencies/include/")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")


    link_directories("${SDK}/lib/")
    
    #cpp-mjpeg-stream(third-part)
    find_package(nadjieb_mjpeg_streamer REQUIRED)

    message("SDK: " ${SDK})

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../dependencies/src/ SRC_FILES)
    add_executable(yolov6_bmcv.soc ${SRC_FILES})
    target_link_libraries(yolov6_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} -lpthread -lavcodec -lavformat -lavutil nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)
    # target_link_libraries(yolov6_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} -lpthread -lavcodec -lavformat -lavutil)
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()