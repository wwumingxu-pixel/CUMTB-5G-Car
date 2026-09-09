// xunxian2 图片巡线测试
// 读取一张图片后持续显示处理结果，按 q、ESC 或关闭窗口退出。
// 不使用摄像头、GPIO、电机或舵机。

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
    constexpr int x_middle = 160;
    constexpr int up_scan_wide = 130;
    constexpr int down_scan_wide = 170;

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

    vector<Point> vertices = {
        Point(0, rows),
        Point(0, static_cast<int>(rows * 0.7)),
        Point(static_cast<int>(cols * 0.45), static_cast<int>(rows * 0.5)),
        Point(static_cast<int>(cols * 0.55), static_cast<int>(rows * 0.5)),
        Point(cols, static_cast<int>(rows * 0.7)),
        Point(cols, rows)
    };
    fillPoly(mask, vector<vector<Point>>{vertices}, Scalar(255));

    Mat selected;
    bitwise_and(image, mask, selected);
    return selected;
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
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return regionSelection(mask);
}

static vector<Point2f> getLineParameters(const vector<Vec4i>& lines) {
    vector<Point2f> parameters;
    parameters.reserve(lines.size());

    for (const auto& detected : lines) {
        float dx = static_cast<float>(detected[2] - detected[0]);
        float dy = static_cast<float>(detected[3] - detected[1]);
        if (fabs(dx) < 1e-5f) {
            continue;
        }

        float slope = dy / dx;
        float intercept = static_cast<float>(detected[1]) - slope * detected[0];
        parameters.emplace_back(slope, intercept);
    }
    return parameters;
}

static double calculateError(const Mat& image, Mat& debug) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0.5, 0.5);

    Mat edges;
    Canny(gray, edges, TrackConfig::canny_low, TrackConfig::canny_high, 3);
    Mat roi_edges = regionSelection(edges);

    vector<Vec4i> lines;
    HoughLinesP(
        roi_edges,
        lines,
        TrackConfig::hough_rho,
        TrackConfig::hough_theta,
        TrackConfig::hough_threshold,
        TrackConfig::hough_min_length,
        TrackConfig::hough_max_gap
    );

    vector<Point2f> parameters = getLineParameters(lines);
    double left_slope = -0.0000001;
    double right_slope = 0.0000001;
    double left_intercept = 0.0;
    double right_intercept = 0.0;
    bool left_found = false;
    bool right_found = false;

    for (const auto& parameter : parameters) {
        double slope = parameter.x;
        double intercept = parameter.y;
        if (slope > 0.25 && slope < 2.0) {
            if (!right_found || slope > right_slope) {
                right_slope = slope;
                right_intercept = intercept;
                right_found = true;
            }
        } else if (slope < -0.25 && slope > -2.0) {
            if (!left_found || slope < left_slope) {
                left_slope = slope;
                left_intercept = intercept;
                left_found = true;
            }
        }
    }

    double middle_sum = 0.0;
    int sample_count = 0;

    for (int y = TrackConfig::up_scan_wide; y < TrackConfig::down_scan_wide; ++y) {
        int left_x = left_found
            ? static_cast<int>((y - left_intercept) / left_slope)
            : 0;
        int right_x = right_found
            ? static_cast<int>((y - right_intercept) / right_slope)
            : image.cols;

        left_x = max(0, min(image.cols - 1, left_x));
        right_x = max(0, min(image.cols - 1, right_x));
        int middle_x = (left_x + right_x) / 2;

        middle_sum += middle_x;
        ++sample_count;

        circle(debug, Point(left_x, y), 2, Scalar(0, 0, 255), -1);
        circle(debug, Point(right_x, y), 2, Scalar(0, 0, 255), -1);
        circle(debug, Point(middle_x, y), 1, Scalar(0, 255, 255), -1);
    }

    line(debug, Point(TrackConfig::x_middle, 0),
         Point(TrackConfig::x_middle, image.rows), Scalar(255, 255, 0), 1, LINE_AA);

    if (sample_count == 0) {
        return 0.0;
    }
    return middle_sum / sample_count - TrackConfig::x_middle;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "用法: ./xunxian2_image_test image.jpg" << endl;
        return 1;
    }

    Mat source = imread(argv[1], IMREAD_COLOR);
    if (source.empty()) {
        cerr << "图片读取失败: " << argv[1] << endl;
        return 1;
    }

    Mat image;
    resize(source, image, Size(TrackConfig::image_width, TrackConfig::image_height));

    namedWindow("xunxian2_image_test", WINDOW_NORMAL);
    resizeWindow("xunxian2_image_test", 1200, 420);

    cout << "xunxian2 图片测试已启动" << endl;
    cout << "图片会持续显示，按 q、ESC 或关闭窗口退出" << endl;

    while (true) {
        Mat white_mask = getWhiteMask(image);
        Mat debug = image.clone();
        double error = calculateError(image, debug);

        Mat mask_bgr;
        cvtColor(white_mask, mask_bgr, COLOR_GRAY2BGR);

        putText(debug, "error=" + to_string(static_cast<int>(error)),
                Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2, LINE_AA);

        Mat raw = image.clone();
        putText(raw, "RAW", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 0), 2, LINE_AA);
        putText(mask_bgr, "WHITE MASK", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 0), 2, LINE_AA);
        putText(debug, "HOUGH + ERROR", Point(10, 48), FONT_HERSHEY_SIMPLEX, 0.55,
                Scalar(0, 255, 0), 1, LINE_AA);

        Mat combined;
        hconcat(vector<Mat>{raw, mask_bgr, debug}, combined);
        imshow("xunxian2_image_test", combined);

        cout << "error=" << static_cast<int>(error) << "\r" << flush;

        int key = waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    cout << endl << "图片测试结束" << endl;
    destroyAllWindows();
    return 0;
}
