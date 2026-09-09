// 单独测试图片的巡线代码
// 作用：读取一张赛道图片，做白线检测并输出误差
// 不依赖摄像头、不依赖 GPIO、电机、舵机

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

namespace TrackConfig {
    constexpr int image_width = 320;
    constexpr int image_height = 180;

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

    Mat masked;
    bitwise_and(image, mask, masked);
    return masked;
}

static Mat getWhiteMask(const Mat& image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    Mat mask;
    inRange(hsv,
           Scalar(TrackConfig::white_h_min, TrackConfig::white_s_min, TrackConfig::white_v_min),
           Scalar(TrackConfig::white_h_max, TrackConfig::white_s_max, TrackConfig::white_v_max),
           mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(mask, mask, kernel);
    erode(mask, mask, kernel);

    return regionSelection(mask);
}

static int computeLineError(const Mat& mask) {
    int error = 0;
    int count = 0;

    int left_line_x = TrackConfig::x_middle;
    int right_line_x = TrackConfig::x_middle;

    for (int y = TrackConfig::up_scan_wide; y <= TrackConfig::down_scan_wide; y += TrackConfig::y_scan_wide) {
        Mat row = mask.row(y);
        int t = TrackConfig::x_middle;

        while (t > TrackConfig::wide_scan + 1) {
            Mat slice = row.colRange(t - TrackConfig::wide_scan, t);
            if (countNonZero(slice) > TrackConfig::wide_need) {
                break;
            }
            t--;
        }
        left_line_x = t;

        t = TrackConfig::x_middle;
        while (t < mask.cols - TrackConfig::wide_scan - 1) {
            Mat slice = row.colRange(t, t + TrackConfig::wide_scan);
            if (countNonZero(slice) > TrackConfig::wide_need) {
                break;
            }
            t++;
        }
        right_line_x = t;

        int middle = (left_line_x + right_line_x) / 2;
        error += middle - TrackConfig::x_middle;
        count++;
    }

    if (count == 0) return 0;
    return error / count;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "用法: ./xunxian_image_test image.jpg" << endl;
        return 1;
    }

    Mat frame = imread(argv[1], IMREAD_COLOR);
    if (frame.empty()) {
        cerr << "图片读取失败: " << argv[1] << endl;
        return 1;
    }

    Mat resized;
    resize(frame, resized, Size(TrackConfig::image_width, TrackConfig::image_height));

    Mat white_mask = getWhiteMask(resized);
    int error = computeLineError(white_mask);

    Mat mask_bgr;
    cvtColor(white_mask, mask_bgr, COLOR_GRAY2BGR);

    Mat display = resized.clone();
    for (int y = 0; y < display.rows; ++y) {
        for (int x = 0; x < display.cols; ++x) {
            if (white_mask.at<uchar>(y, x) > 0) {
                display.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }

    line(display, Point(TrackConfig::x_middle, 0), Point(TrackConfig::x_middle, display.rows), Scalar(255, 255, 0), 1);

    putText(display, "error=" + to_string(error), Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    namedWindow("image_test", WINDOW_NORMAL);
    resizeWindow("image_test", 1200, 420);

    Mat combined;
    hconcat(vector<Mat>{resized, mask_bgr, display}, combined);
    imshow("image_test", combined);

    cout << "图片巡线测试完成" << endl;
    cout << "误差 error = " << error << endl;
    waitKey(0);

    destroyAllWindows();
    return 0;
}
