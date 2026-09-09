// 纯视觉巡线测试
// 移植自 run_cpp.cpp 中的巡线扫描逻辑
// 不包含 GPIO、电机或舵机控制，只保留图像处理与显示。

#include <opencv2/opencv.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace cv;
using namespace std;

namespace TrackConfig {
    constexpr int image_width = 320;
    constexpr int image_height = 180;
    constexpr int camera_index = 0;

    // 赛道/边线提取参数，沿用 run_cpp.cpp 的思路
    constexpr int x_middle = 160;
    constexpr int wide_scan = 5;
    constexpr int wide_need = 2;
    constexpr int up_scan_wide = 130;
    constexpr int down_scan_wide = 170;
    constexpr int y_scan_wide = 5;

    constexpr double kp = 0.3;
    constexpr double kd = 2.0;
    constexpr int all_count = (down_scan_wide - up_scan_wide) / y_scan_wide;

    // 红色赛道分割阈值
    constexpr int red1_h_min = 0;
    constexpr int red1_h_max = 34;
    constexpr int red2_h_min = 160;
    constexpr int red2_h_max = 179;
    constexpr int red_s_min = 75;
    constexpr int red_v_min = 70;

    // 蓝色掩码（可用于筛选非赛道噪声）
    constexpr int blue_h_min = 100;
    constexpr int blue_h_max = 140;
    constexpr int blue_s_min = 37;
    constexpr int blue_v_min = 100;

    // 霍夫/边缘仅保留用于显示，不参与主巡线逻辑
    constexpr int canny_low = 70;
    constexpr int canny_high = 150;
}

struct TrackState {
    int error = 0;
    int last_error = 0;
    std::array<uint16_t, 10> draw_middle_line_x{};
    std::array<uint16_t, 10> draw_left_line_x{};
    std::array<uint16_t, 10> draw_right_line_x{};
};

static TrackState g_track;

static Mat processTrackMask(const Mat& frame) {
    Mat resized;
    resize(frame, resized, Size(TrackConfig::image_width, TrackConfig::image_height));

    Mat blur;
    GaussianBlur(resized, blur, Size(5, 5), 0);

    Mat hsv;
    cvtColor(blur, hsv, COLOR_BGR2HSV);

    Mat mask1, mask2, mask3;
    inRange(hsv,
           Scalar(TrackConfig::red1_h_min, TrackConfig::red_s_min, TrackConfig::red_v_min),
           Scalar(TrackConfig::red1_h_max, 255, 255),
           mask1);
    inRange(hsv,
           Scalar(TrackConfig::red2_h_min, TrackConfig::red_s_min, TrackConfig::red_v_min),
           Scalar(TrackConfig::red2_h_max, 255, 255),
           mask2);
    inRange(hsv,
           Scalar(TrackConfig::blue_h_min, TrackConfig::blue_s_min, TrackConfig::blue_v_min),
           Scalar(TrackConfig::blue_h_max, 255, 255),
           mask3);

    Mat combined = mask1 | mask2 | mask3;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(combined, combined, kernel, Point(-1, -1), 2);
    morphologyEx(combined, combined, MORPH_OPEN, kernel);
    morphologyEx(combined, combined, MORPH_CLOSE, kernel);

    return combined;
}

static void lineWalking(const Mat& frame_final) {
    g_track.error = 0;
    Mat scan_y_wide = frame_final(Range(TrackConfig::up_scan_wide, TrackConfig::down_scan_wide), Range::all());

    int theory_middle_line = TrackConfig::x_middle;
    int left_line_x = TrackConfig::x_middle;
    int right_line_x = TrackConfig::x_middle;

    g_track.draw_middle_line_x[0] = TrackConfig::x_middle;

    int i = 1;

    Mat y_line = scan_y_wide.row(scan_y_wide.rows - 1);
    while (left_line_x > TrackConfig::wide_scan + 1) {
        Mat wide_range = y_line(Range::all(), Range(left_line_x - TrackConfig::wide_scan, left_line_x));
        if (countNonZero(wide_range) > TrackConfig::wide_need) {
            break;
        }
        left_line_x--;
    }

    while (right_line_x < TrackConfig::image_width - TrackConfig::wide_scan - 1) {
        Mat wide_range = y_line(Range::all(), Range(right_line_x, right_line_x + TrackConfig::wide_scan));
        if (countNonZero(wide_range) > TrackConfig::wide_need) {
            break;
        }
        right_line_x++;
    }

    g_track.draw_left_line_x[0] = left_line_x;
    g_track.draw_right_line_x[0] = right_line_x;
    g_track.draw_middle_line_x[0] = (right_line_x + left_line_x) / 2;

    for (int y = scan_y_wide.rows - 1; y >= 0; y -= TrackConfig::y_scan_wide) {
        y_line = scan_y_wide.row(y);
        int t = g_track.draw_middle_line_x[i - 1];

        while (t > TrackConfig::wide_scan + 1) {
            Mat wide_range = y_line(Range::all(), Range(t - TrackConfig::wide_scan, t));
            if (countNonZero(wide_range) > TrackConfig::wide_need) {
                break;
            }
            left_line_x = t;
            t--;
        }

        t = g_track.draw_middle_line_x[i - 1];
        while (t < TrackConfig::image_width - TrackConfig::wide_scan - 1) {
            Mat wide_range = y_line(Range::all(), Range(t, t + TrackConfig::wide_scan));
            if (countNonZero(wide_range) > TrackConfig::wide_need) {
                break;
            }
            right_line_x = t;
            t++;
        }

        g_track.draw_middle_line_x[i] = (right_line_x + left_line_x) / 2;
        g_track.draw_left_line_x[i] = left_line_x;
        g_track.draw_right_line_x[i] = right_line_x;
        g_track.error += static_cast<int>(g_track.draw_middle_line_x[i] - theory_middle_line);
        i++;
    }

    g_track.error /= TrackConfig::all_count;
}

static double pidControl(double error) {
    double derivative = error - g_track.last_error;
    g_track.last_error = static_cast<int>(error);
    double output = TrackConfig::kp * error + TrackConfig::kd * derivative;

    if (output > 8.0) return 8.0;
    if (output < -8.0) return -8.0;
    return output;
}

static void drawTrackInfo(Mat& display, const Mat& frame_final) {
    int line_count = 0;
    for (int idx = 0; idx < 10; ++idx) {
        if (g_track.draw_middle_line_x[idx] == 0 && idx > 0) {
            break;
        }
        line_count++;
    }

    for (int idx = 0; idx < line_count; ++idx) {
        int y = TrackConfig::up_scan_wide + idx * TrackConfig::y_scan_wide;
        int x_mid = g_track.draw_middle_line_x[idx];
        int x_left = g_track.draw_left_line_x[idx];
        int x_right = g_track.draw_right_line_x[idx];

        circle(display, Point(x_mid, y), 2, Scalar(0, 255, 255), -1);
        line(display, Point(x_left, y), Point(x_right, y), Scalar(0, 255, 0), 1, LINE_AA);
    }

    line(display, Point(TrackConfig::x_middle, 0), Point(TrackConfig::x_middle, TrackConfig::image_height),
         Scalar(255, 255, 0), 1, LINE_AA);

    putText(display,
            "error=" + to_string(g_track.error),
            Point(10, 20),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(0, 255, 255),
            1,
            LINE_AA);

    putText(display,
            "pid=" + to_string(pidControl(static_cast<double>(g_track.error))),
            Point(10, 40),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(0, 255, 255),
            1,
            LINE_AA);
}

int main() {
    VideoCapture camera(TrackConfig::camera_index, CAP_V4L2);
    if (!camera.isOpened()) {
        cerr << "摄像头打开失败" << endl;
        return 1;
    }

    camera.set(CAP_PROP_FRAME_WIDTH, TrackConfig::image_width);
    camera.set(CAP_PROP_FRAME_HEIGHT, TrackConfig::image_height);
    camera.set(CAP_PROP_FPS, 30);
    camera.set(CAP_PROP_BUFFERSIZE, 1);

    namedWindow("xunxian3_visual", WINDOW_NORMAL);
    resizeWindow("xunxian3_visual", 1200, 420);

    cout << "纯视觉巡线测试启动（移植 run_cpp 巡线逻辑）" << endl;
    cout << "按 ESC / q 退出" << endl;

    while (true) {
        Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            cerr << "读取摄像头图像失败" << endl;
            continue;
        }

        Mat mask = processTrackMask(frame);
        lineWalking(mask);

        Mat raw_bgr;
        resize(frame, raw_bgr, Size(TrackConfig::image_width, TrackConfig::image_height));

        Mat mask_bgr;
        cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);

        Mat overlay = raw_bgr.clone();
        for (int y = 0; y < overlay.rows; ++y) {
            for (int x = 0; x < overlay.cols; ++x) {
                if (mask.at<uchar>(y, x) > 0) {
                    overlay.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
                }
            }
        }

        drawTrackInfo(overlay, mask);

        putText(raw_bgr, "RAW", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        putText(mask_bgr, "MASK", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        putText(overlay, "OVERLAY", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);

        Mat combined;
        hconcat(vector<Mat>{raw_bgr, mask_bgr, overlay}, combined);
        imshow("xunxian3_visual", combined);

        cout << "error=" << g_track.error
             << " pid=" << pidControl(static_cast<double>(g_track.error))
             << endl;

        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    camera.release();
    destroyAllWindows();
    return 0;
}
