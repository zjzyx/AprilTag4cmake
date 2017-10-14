/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <iostream>

#include <opencv2/opencv.hpp>
#include "apriltag.h"
#include "tag36h11.h"
#include "tag36h10.h"
#include "tag36artoolkit.h"
#include "tag25h9.h"
#include "tag25h7.h"
#include "common/getopt.h"

using namespace std;
using namespace cv;

/*
 * 这个apriltag的代码可以当做工具，来获取位姿
 * 这个main.cc和作者提供的opencv_demo.cc的区别就在于我们加上了三个坐标轴
 * 做法:
 *    apriltag的检测会自动帮我们将apriltag的四个边缘上的角找到，
 *    这个det->p数据结构是一个数组，并且里面的0,1,2,3个元素代表的角点是固定的
 *    我们可以定义检测到的角点在世界坐标系下的坐标，我们可以随意选取原点坐标来确定角点在世界坐标系下的坐标
 *    但是一般来说我们都会把这个点选择在这个tag的中心位置。
 *    现在我们来假想一个帧，这个帧具有如下特点:
 *    在程序运行之前就存在，当我们固定好tag和拿好相机的那个刹那就存在;
 *    这个帧的相机坐标系和世界坐标系是一样的;
 *
 *    又根据我们现在检测到的角点在图像坐标系下的坐标，这个和假设帧检测到的角点是对应的
 *    那么我们现在就可以用PNP的方法去求解当前帧的位姿了。
 *
 *    在slam中，我们通常用前后两帧来构造PNP求解的问题，计算的是相对于上一帧的位姿变换，并且用的坐标也是相机坐标系下的坐标，
 *    而在这里，我们的上一帧就是假想帧，且假想帧的坐标就是世界坐标系下的坐标
 *
 *    根据上面我们可以求出相对于假想帧的位姿变换，我们要加的坐标轴，其实就是世界坐标系下的坐标轴经过这个位姿变换，
 *    又经过相机内参变换，投影到当前像素坐标系下的位置
 */

/*
 * The HD Digital Camera's Camera parameters
 */
//double camera_matrix[] =
//{
//    524.97f, 0.0f, 295.29f,
//    0.0f, 525.32f, 210.06f,
//    0.0f, 0.0f, 1.0f
//};
//double dist_coeff[] = {-0.026384f, -0.022035f, -0.005807f, -0.008081f};

/*
 * my laptop camera parameters
 */
double camera_matrix[] =
{
    745.7, 0.0, 337.5,
    0.0, 749.1, 260.3,
    0.0, 0.0, 1.0
};
double dist_coeff[] = {0.297, -1.2, 0.0077, 0.0029};

Mat m_camera_matrix = Mat(3, 3, CV_64FC1, camera_matrix).clone();
Mat m_dist_coeff = Mat(1, 4, CV_64FC1, dist_coeff).clone();

//我们来定义一下那四个角点在世界坐标系下的坐标，这样其实世界坐标轴也固定了
Point3f corners_3d[] =
{
    Point3f(-1.0f, -1.0f, 0),
    Point3f(-1.0f,  1.0f, 0),
    Point3f( 1.0f,  1.0f, 0),
    Point3f( 1.0f, -1.0f, 0)
};
vector<Point3f> m_corners_3d = vector<Point3f>(corners_3d, corners_3d + 4);



int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, '\0', "border", "1", "Set tag family border size");
    getopt_add_int(getopt, 't', "threads", "4", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "1.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
    getopt_add_bool(getopt, '1', "refine-decode", 0, "Spend more time trying to decode tags");
    getopt_add_bool(getopt, '2', "refine-pose", 0, "Spend more time trying to precisely localize tags");

    if (!getopt_parse(getopt, argc, argv, 1) ||
            getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    // Initialize camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Couldn't open video capture device" << endl;
        return -1;
    }

    // Initialize tag detector with options
    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11"))
        tf = tag36h11_create();
    else if (!strcmp(famname, "tag36h10"))
        tf = tag36h10_create();
    else if (!strcmp(famname, "tag36artoolkit"))
        tf = tag36artoolkit_create();
    else if (!strcmp(famname, "tag25h9"))
        tf = tag25h9_create();
    else if (!strcmp(famname, "tag25h7"))
        tf = tag25h7_create();
    else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }
    tf->black_border = getopt_get_int(getopt, "border");

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");
    td->refine_decode = getopt_get_bool(getopt, "refine-decode");
    td->refine_pose = getopt_get_bool(getopt, "refine-pose");

    Mat frame, gray;
    while (true) {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        // 给结构体赋值原来还可以这样子
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };

        zarray_t *detections = apriltag_detector_detect(td, &im);
        cout << zarray_size(detections) << " tags detected" << endl;

        // Draw detection outlines
        // Color channel bgr
        // The order of the points is arranged anticlockwise
        // 点逆时针排序
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(255, 0, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 255, 0), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0, 0, 255), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(255, 255, 255), 2);

            stringstream ss;
            ss << det->id;
            String text = ss.str();
            int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontscale = 1;
            int baseline;
            Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,
                                       det->c[1]+textsize.height/2),
                    fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

            //------------------- show Three-dimensional coordinate system ---------
            double axisfontscale = 0.5;
            Mat r, t;
            // 当前帧的二维坐标
            vector<Point2f> m_corners;
            m_corners.push_back(Point2f(det->p[0][0], det->p[0][1]));
            m_corners.push_back(Point2f(det->p[1][0], det->p[1][1]));
            m_corners.push_back(Point2f(det->p[2][0], det->p[2][1]));
            m_corners.push_back(Point2f(det->p[3][0], det->p[3][1]));
            Mat rot_vec;
            // pnp求解位姿
            solvePnP(m_corners_3d, m_corners, m_camera_matrix, m_dist_coeff, rot_vec, t);
            Rodrigues(rot_vec, r);

            // 世界坐标轴上的点
            vector< Point3f > axisPoints;
            axisPoints.push_back(Point3f(0, 0, 0));
            axisPoints.push_back(Point3f(1, 0, 0));
            axisPoints.push_back(Point3f(0, 1, 0));
            axisPoints.push_back(Point3f(0, 0, -1));
            vector< Point2f > imagePoints;
            // 投影到当前像素坐标下的点
            projectPoints(axisPoints, r, t, m_camera_matrix, m_dist_coeff, imagePoints);

            // draw axis lines and write coordinate axis information
            line(frame, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 2);
            putText(frame, "X", imagePoints[1],
                    FONT_HERSHEY_SIMPLEX, axisfontscale, Scalar(0xff, 0x99, 0), 2);
            line(frame, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 2);
            putText(frame, "Y", imagePoints[2],
                    FONT_HERSHEY_SIMPLEX, axisfontscale, Scalar(0xff, 0x99, 0), 2);
            line(frame, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 2);
            putText(frame, "Z", imagePoints[3],
                    FONT_HERSHEY_SIMPLEX, axisfontscale, Scalar(0xff, 0x99, 0), 2);

            //------------------- show Three-dimensional coordinate system ---------
        }
        zarray_destroy(detections);

        imshow("Tag Detections", frame);
        if (waitKey(30) >= 0)
            break;
    }

    apriltag_detector_destroy(td);
    if (!strcmp(famname, "tag36h11"))
        tag36h11_destroy(tf);
    else if (!strcmp(famname, "tag36h10"))
        tag36h10_destroy(tf);
    else if (!strcmp(famname, "tag36artoolkit"))
        tag36artoolkit_destroy(tf);
    else if (!strcmp(famname, "tag25h9"))
        tag25h9_destroy(tf);
    else if (!strcmp(famname, "tag25h7"))
        tag25h7_destroy(tf);
    getopt_destroy(getopt);

    return 0;
}
