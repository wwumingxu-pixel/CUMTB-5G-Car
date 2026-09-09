// xunxian3 图片巡线测试
// 读取一张图片后持续显示结果，按 q 或 ESC 退出。
// 不使用摄像头、GPIO、电机或舵机。

#include <opencv2/opencv.hpp>

#include <array>
#include <iostream>
#include <string>

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

    constexpr int red1_h_min = 0;
    constexpr int red1_h_max = 34;
    constexpr int red2_h_min = 160;
    constexpr int red2_h_max = 179;
    constexpr int red_s_min = 75;
    constexpr int red_v_min = 70;

    constexpr int blue_h_min = 100;
    constexpr int blue_h_max = 140;
    constexpr int blue_s_min = 37;
    constexpr int blue_v_min = 100;
}

struct ScanState {
    int error = 0;
    std::array<int, 10> middle{};
    std::array<int, 10> left{};
    std::array<int, 10> right{};
};

static ScanState scan_state;

static Mat processTrackMask(const Mat& frame) {
    Mat resized;
    resize(frame, resized, Size(TrackConfig::image_width, TrackConfig::image_height));

    Mat blurred;
    GaussianBlur(resized, blurred, Size(5, 5), 0);

    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    Mat red_mask1;
    Mat red_mask2;
    Mat blue_mask;
    inRange(hsv,
            Scalar(TrackConfig::red1_h_min, TrackConfig::red_s_min, TrackConfig::red_v_min),
            Scalar(TrackConfig::red1_h_max, 255, 255), red_mask1);
    inRange(hsv,
            Scalar(TrackConfig::red2_h_min, TrackConfig::red_s_min, TrackConfig::red_v_min),
            Scalar(TrackConfig::red2_h_max, 255, 255), red_mask2);
    inRange(hsv,
            Scalar(TrackConfig::blue_h_min, TrackConfig::blue_s_min, TrackConfig::blue_v_min),
            Scalar(TrackConfig::blue_h_max, 255, 255), blue_mask);

    Mat mask = red_mask1 | red_mask2 | blue_mask;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(mask, mask, kernel, Point(-1, -1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

static void scanLine(const Mat& mask) {
    scan_state = ScanState{};
    int left_line_x = TrackConfig::x_middle;
    int right_line_x = TrackConfig::x_middle;
    int sample_count = 0;

    Mat scan_area = mask(Range(TrackConfig::up_scan_wide, TrackConfig::down_scan_wide), Range::all());
    Mat bottom_row = scan_area.row(scan_area.rows - 1);

    while (left_line_x > TrackConfig::wide_scan + 1) {
        Mat area = bottom_row.colRange(left_line_x - TrackConfig::wide_scan, left_line_x);
        if (countNonZero(area) > TrackConfig::wide_need) break;
        --left_line_x;
    }
    while (right_line_x < TrackConfig::image_width - TrackConfig::wide_scan - 1) {
        Mat area = bottom_row.colRange(right_line_x, right_line_x + TrackConfig::wide_scan);
        if (countNonZero(area) > TrackConfig::wide_need) break;
        ++right_line_x;
    }

    scan_state.left[0] = left_line_x;
    scan_state.right[0] = right_line_x;
    scan_state.middle[0] = (left_line_x + right_line_x) / 2;
    scan_state.error = scan_state.middle[0] - TrackConfig::x_middle;
    sample_count = 1;

    int index = 1;
    for (int row = scan_area.rows - 1 - TrackConfig::y_scan_wide;
         row >= 0 && index < static_cast<int>(scan_state.middle.size());
         row -= TrackConfig::y_scan_wide) {
        Mat current_row = scan_area.row(row);
        int center = scan_state.middle[index - 1];
        int candidate = center;

        while (candidate > TrackConfig::wide_scan + 1) {
            Mat area = current_row.colRange(candidate - TrackConfig::wide_scan, candidate);
            if (countNonZero(area) > TrackConfig::wide_need) break;
            left_line_x = candidate;
            --candidate;
        }

        candidate = center;
        while (candidate < TrackConfig::image_width - TrackConfig::wide_scan - 1) {
            Mat area = current_row.colRange(candidate, candidate + TrackConfig::wide_scan);
            if (countNonZero(area) > TrackConfig::wide_need) break;
            right_line_x = candidate;
            ++candidate;
        }

        scan_state.left[index] = left_line_x;
        scan_state.right[index] = right_line_x;
        scan_state.middle[index] = (left_line_x + right_line_x) / 2;
        scan_state.error += scan_state.middle[index] - TrackConfig::x_middle;
        ++sample_count;
        ++index;
    }

    if (sample_count > 0) {
        scan_state.error /= sample_count;
    }
}

static void drawResult(Mat& display) {
    for (int index = 0; index < 10; ++index) {
        int y = TrackConfig::down_scan_wide - index * TrackConfig::y_scan_wide;
        if (y < TrackConfig::up_scan_wide) break;

        circle(display, Point(scan_state.left[index], y), 2, Scalar(0, 0, 255), -1);
        circle(display, Point(scan_state.right[index], y), 2, Scalar(0, 0, 255), -1);
        circle(display, Point(scan_state.middle[index], y), 2, Scalar(0, 255, 255), -1);
        line(display,
             Point(scan_state.left[index], y),
             Point(scan_state.right[index], y),
             Scalar(0, 255, 0), 1, LINE_AA);
    }

    line(display, Point(TrackConfig::x_middle, 0),
         Point(TrackConfig::x_middle, display.rows), Scalar(255, 255, 0), 1, LINE_AA);
    putText(display, "error=" + to_string(scan_state.error), Point(10, 25),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2, LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "用法: ./xunxian3_image_test image.jpg" << endl;
        return 1;
    }

    Mat source = imread(argv[1], IMREAD_COLOR);
    if (source.empty()) {
        cerr << "图片读取失败: " << argv[1] << endl;
        return 1;
    }

    Mat image;
    resize(source, image, Size(TrackConfig::image_width, TrackConfig::image_height));

    namedWindow("xunxian3_image_test", WINDOW_NORMAL);
    resizeWindow("xunxian3_image_test", 1200, 420);
    cout << "xunxian3 图片测试启动，按 q 或 ESC 退出" << endl;

    while (true) {
        Mat mask = processTrackMask(image);
        scanLine(mask);

        Mat raw = image.clone();
        Mat mask_bgr;
        cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);
        Mat result = image.clone();
        drawResult(result);

        putText(raw, "RAW", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 0), 2, LINE_AA);
        putText(mask_bgr, "TRACK MASK", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 0), 2, LINE_AA);
        putText(result, "SCAN RESULT", Point(10, 48), FONT_HERSHEY_SIMPLEX, 0.55,
                Scalar(0, 255, 0), 1, LINE_AA);

        Mat combined;
        hconcat(vector<Mat>{raw, mask_bgr, result}, combined);
        imshow("xunxian3_image_test", combined);
        cout << "error=" << scan_state.error << "\r" << flush;

        int key = waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cout << endl << "图片测试结束" << endl;
    destroyAllWindows();
    return 0;
}
