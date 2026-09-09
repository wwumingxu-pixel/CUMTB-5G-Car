// 纯视觉巡线测试
// 移植自 simple_code.cpp 中的赛道边界扫描思路
// 不包含 GPIO、电机、舵机控制，只保留图像处理与显示。

#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

namespace TrackConfig {
    constexpr int image_width = 320;
    constexpr int image_height = 180;
    constexpr int camera_index = 0;

    constexpr int x_middle = 160;
    constexpr int wide_scan = 5;
    constexpr int wide_need = 2;
    constexpr int up_scan_wide = 130;
    constexpr int down_scan_wide = 170;
    constexpr int y_scan_wide = 5;

    constexpr int white_h_min = 0;
    constexpr int white_h_max = 180;
    constexpr int white_s_min = 0;
    constexpr int white_s_max = 50;
    constexpr int white_v_min = 180;
    constexpr int white_v_max = 255;

    constexpr int canny_low = 60;
    constexpr int canny_high = 140;
    constexpr double hough_rho = 1.0;
    constexpr double hough_theta = CV_PI / 180.0;
    constexpr int hough_threshold = 50;
    constexpr int hough_min_length = 30;
    constexpr int hough_max_gap = 5;
}

static Mat regionSelection(const Mat& image) {
    Mat mask = Mat::zeros(image.size(), image.type());
    int rows = image.rows;
    int cols = image.cols;

    Point bottom_left(0, rows);
    Point mid_left(0, static_cast<int>(rows * 0.7));
    Point top_left(static_cast<int>(cols * 0.45), static_cast<int>(rows * 0.5));
    Point top_right(static_cast<int>(cols * 0.55), static_cast<int>(rows * 0.5));
    Point mid_right(cols, static_cast<int>(rows * 0.7));
    Point bottom_right(cols, rows);

    vector<Point> vertices = {bottom_left, mid_left, top_left, top_right, mid_right, bottom_right};
    vector<vector<Point>> pts = {vertices};
    fillPoly(mask, pts, Scalar(255));

    Mat masked_image;
    bitwise_and(image, mask, masked_image);
    return masked_image;
}

static Mat getWhiteMask(const Mat& image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    Mat mask;
    inRange(
        hsv,
        Scalar(TrackConfig::white_h_min, TrackConfig::white_s_min, TrackConfig::white_v_min),
        Scalar(TrackConfig::white_h_max, TrackConfig::white_s_max, TrackConfig::white_v_max),
        mask
    );

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(mask, mask, kernel);
    erode(mask, mask, kernel);

    return regionSelection(mask);
}

static vector<Point2f> getLinesFangcheng(const vector<Vec4i>& lines) {
    vector<Point2f> output;
    output.reserve(lines.size());

    for (const auto& line : lines) {
        float x1 = static_cast<float>(line[0]);
        float y1 = static_cast<float>(line[1]);
        float x2 = static_cast<float>(line[2]);
        float y2 = static_cast<float>(line[3]);

        float k = (y2 - y1) / (x2 - x1 + 1e-6f);
        float b = y1 - k * x1;
        output.emplace_back(k, b);
    }
    return output;
}

static double computeLineError(const Mat& frame, const Mat& mask, Mat& debugFrame) {
    Mat canny;
    Mat grayed;
    cvtColor(frame, grayed, COLOR_BGR2GRAY);
    GaussianBlur(grayed, grayed, Size(5, 5), 0);
    Canny(grayed, canny, TrackConfig::canny_low, TrackConfig::canny_high, 3);

    vector<Vec4i> lines;
    HoughLinesP(canny, lines, TrackConfig::hough_rho, TrackConfig::hough_theta,
                TrackConfig::hough_threshold, TrackConfig::hough_min_length,
                TrackConfig::hough_max_gap);

    if (lines.empty()) {
        return 0.0;
    }

    vector<Point2f> line_params = getLinesFangcheng(lines);

    double kr = 0.00000001;
    double kl = -0.0000001;
    double br = 0.0;
    double bl = 0.0;
    int lr = 0;
    int ll = 0;

    for (size_t i = 0; i < line_params.size(); ++i) {
        float k = line_params[i].x;
        if ((k > 0.25 && k < 2.0) || (k < -0.25 && k > -2.0)) {
            if (k > 0) {
                ++lr;
                if (k > kr) {
                    kr = k;
                    br = line_params[i].y;
                }
            } else if (k < 0) {
                ++ll;
                if (k < kl) {
                    kl = k;
                    bl = line_params[i].y;
                }
            }
        }
    }

    int flagl = 0;
    int flagr = 0;
    if (lr == 0) {
        flagr = 1;
    }
    if (ll == 0) {
        flagl = 1;
    }

    double ave_x = 0.0;
    int count = 0;

    for (int y = TrackConfig::up_scan_wide; y < TrackConfig::down_scan_wide; ++y) {
        int l, r;
        if (flagl) {
            l = 0;
        } else {
            l = static_cast<int>((y - bl) / kl);
        }
        if (flagr) {
            r = debugFrame.cols;
        } else {
            r = static_cast<int>((y - br) / kr);
        }

        double mid = (l + r) / 2.0;
        ave_x += mid;
        ++count;

        circle(debugFrame, Point(r, y), 2, Scalar(0, 0, 255), -1);
        circle(debugFrame, Point(l, y), 2, Scalar(0, 0, 255), -1);
        circle(debugFrame, Point((l + r) / 2, y), 1, Scalar(0, 255, 255), -1);
    }

    if (count == 0) {
        return 0.0;
    }

    ave_x /= count;
    double error = ave_x - (frame.cols / 2.0);

    line(debugFrame, Point(TrackConfig::x_middle, 0), Point(TrackConfig::x_middle, debugFrame.rows),
         Scalar(255, 255, 0), 1, LINE_AA);
    putText(debugFrame, "error=" + to_string(static_cast<int>(error)),
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2, LINE_AA);

    return error;
}

int main() {
    VideoCapture camera(TrackConfig::camera_index, CAP_V4L2);
    if (!camera.isOpened()) {
        cerr << "摄像头打开失败" << endl;
        return 1;
    }

    camera.set(CAP_PROP_FRAME_WIDTH, TrackConfig::image_width);
    camera.set(CAP_PROP_FRAME_HEIGHT, TrackConfig::image_height);
    camera.set(CAP_PROP_BUFFERSIZE, 1);

    namedWindow("xunxian2_visual", WINDOW_NORMAL);
    resizeWindow("xunxian2_visual", 1200, 420);

    cout << "移植 simple_code 巡线逻辑启动" << endl;
    cout << "按 ESC / q 退出" << endl;

    while (true) {
        Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            cerr << "读取摄像头失败" << endl;
            continue;
        }

        Mat resized;
        resize(frame, resized, Size(TrackConfig::image_width, TrackConfig::image_height));

        Mat white_mask = getWhiteMask(resized);

        Mat debugFrame = resized.clone();
        double error = computeLineError(resized, white_mask, debugFrame);

        Mat white_bgr;
        cvtColor(white_mask, white_bgr, COLOR_GRAY2BGR);

        putText(resized, "RAW", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        putText(white_bgr, "WHITE MASK", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        putText(debugFrame, "LINE ERROR", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);

        Mat combined;
        hconcat(vector<Mat>{resized, white_bgr, debugFrame}, combined);
        imshow("xunxian2_visual", combined);

        cout << "error=" << static_cast<int>(error) << endl;

        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    camera.release();
    destroyAllWindows();
    return 0;
}
