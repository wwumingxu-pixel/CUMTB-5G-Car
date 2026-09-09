// 纯视觉跑道白线检测测试
// 移植自 smartCar-main/g5g-new/paodao.cpp
// 不包含 GPIO、电机、舵机控制，只做摄像头图像处理与显示。

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
	constexpr int camera_index = -1;

	// HSV 白色阈值：低饱和、高亮度。
	constexpr int white_h_min = 0;
	constexpr int white_h_max = 180;
	constexpr int white_s_min = 0;
	constexpr int white_s_max = 50;
	constexpr int white_v_min = 180;
	constexpr int white_v_max = 255;

	// Canny + 霍夫直线参数。
	constexpr int canny_low = 70;
	constexpr int canny_high = 150;
	constexpr int hough_rho = 2;
	constexpr int hough_threshold = 20;
	constexpr int hough_min_line_length = 70;
	constexpr int hough_max_line_gap = 20;

	// 只保留有明显斜率的线段，过滤横线和接近竖直的杂线。
	constexpr double line_angle_min = 20.0;
	constexpr double line_angle_max = 70.0;
}

static Mat regionSelection(const Mat& image) {
	Mat mask = Mat::zeros(image.size(), image.type());
	int channel_count = image.channels();
	Scalar ignore_mask_color = (channel_count == 3) ? Scalar(255, 255, 255) : Scalar(255);

	int rows = image.rows;
	int cols = image.cols;
	Point bottom_left(0, rows);
	Point top_left(static_cast<int>(cols * 0.45), static_cast<int>(rows * 0.5));
	Point top_right(static_cast<int>(cols * 0.55), static_cast<int>(rows * 0.5));
	Point bottom_right(cols, rows);
	Point mid_left(0, static_cast<int>(rows * 0.7));
	Point mid_right(cols, static_cast<int>(rows * 0.7));

	vector<Point> vertices = {bottom_left, mid_left, top_left, top_right, mid_right, bottom_right};
	vector<vector<Point>> pts = {vertices};
	fillPoly(mask, pts, ignore_mask_color);

	Mat masked_image;
	bitwise_and(image, mask, masked_image);
	return masked_image;
}

static vector<Vec4i> detectLines(const Mat& image) {
	vector<Vec4i> temp;
	HoughLinesP(
		image,
		temp,
		TrackConfig::hough_rho,
		CV_PI / 180,
		TrackConfig::hough_threshold,
		TrackConfig::hough_min_line_length,
		TrackConfig::hough_max_line_gap
	);

	vector<Vec4i> lines;
	for (const auto& line : temp) {
		double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
		bool positive = angle < TrackConfig::line_angle_max && angle > TrackConfig::line_angle_min;
		bool negative = angle < -TrackConfig::line_angle_min && angle > -TrackConfig::line_angle_max;
		if (positive || negative) {
			lines.push_back(line);
		}
	}
	return lines;
}

static void drawLinesOnBinaryImage(const vector<Vec4i>& lines, Mat& binary_image) {
	for (const Vec4i& detected_line : lines) {
		Point pt1(detected_line[0], detected_line[1]);
		Point pt2(detected_line[2], detected_line[3]);
		line(binary_image, pt1, pt2, Scalar(255), 2, LINE_AA);
	}
}

// Canny 边缘 + 霍夫直线方法。
static Mat frameProcessorByHF(const Mat& image) {
	Mat grayscale;
	cvtColor(image, grayscale, COLOR_BGR2GRAY);

	Mat blurred;
	GaussianBlur(grayscale, blurred, Size(5, 5), 0);

	Mat edges;
	Canny(blurred, edges, TrackConfig::canny_low, TrackConfig::canny_high);

	Mat region = regionSelection(edges);
	vector<Vec4i> hough = detectLines(region);

	Mat binary_image = Mat::zeros(image.size(), CV_8UC1);
	drawLinesOnBinaryImage(hough, binary_image);
	return binary_image;
}

// HSV 白色二值化方法，适合红色操场中间白线。
static Mat frameProcessorByEZ(const Mat& image) {
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

	mask = regionSelection(mask);
	return mask;
}

int main() {
	VideoCapture camera(TrackConfig::camera_index);
	if (!camera.isOpened()) {
		cerr << "摄像头打开失败" << endl;
		return 1;
	}

	camera.set(CAP_PROP_FRAME_WIDTH, TrackConfig::image_width);
	camera.set(CAP_PROP_FRAME_HEIGHT, TrackConfig::image_height);
	camera.set(CAP_PROP_BUFFERSIZE, 1);

	namedWindow("xunxian_paodao", WINDOW_NORMAL);
	resizeWindow("xunxian_paodao", 960, 540);

	cout << "纯视觉白线检测启动" << endl;
	cout << "左：原图  中：白线二值化  右：霍夫直线" << endl;
	cout << "按 ESC / q 退出" << endl;

	while (true) {
		Mat frame;
		camera >> frame;
		if (frame.empty()) {
			cerr << "读取摄像头画面失败" << endl;
			continue;
		}

		Mat resized;
		resize(frame, resized, Size(TrackConfig::image_width, TrackConfig::image_height));

		Mat white_mask = frameProcessorByEZ(resized);
		Mat hough_mask = frameProcessorByHF(resized);

		Mat white_bgr;
		Mat hough_bgr;
		cvtColor(white_mask, white_bgr, COLOR_GRAY2BGR);
		cvtColor(hough_mask, hough_bgr, COLOR_GRAY2BGR);

		putText(resized, "RAW", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
		putText(white_bgr, "WHITE MASK", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
		putText(hough_bgr, "HOUGH LINE", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);

		Mat combined;
		hconcat(vector<Mat>{resized, white_bgr, hough_bgr}, combined);
		imshow("xunxian_paodao", combined);

		int key = waitKey(1);
		if (key == 27 || key == 'q' || key == 'Q') {
			break;
		}
	}

	camera.release();
	destroyAllWindows();
	return 0;
}
