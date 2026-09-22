// 斑马线纯视觉检测测试
// 算法参考：smartCar-main/g5g-new/crossroad.cpp
// 更新说明：摄像头改为先采集 1920x1080，再缩放到 320x180 做斑马线检测，保留更完整视野。
// 更新日期：2026-09

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

namespace ZebraConfig {
	constexpr int capture_width = 1920;
	constexpr int capture_height = 1080;
	constexpr int image_width = 320;
	constexpr int image_height = 180;

	// OTSU 阈值下限。过暗场景中自动阈值过低会令跑道大面积变白。
	constexpr double otsu_threshold_min = 100.0;

	// 视野：整张图的下半部分。
	constexpr double roi_top_ratio = 0.55;

	// 白色色块判断：至少找到 4 个面积足够的白色色块。
	constexpr int required_blocks = 4;
	constexpr int block_min_area = 65;
	constexpr int block_min_width = 5;
	constexpr int block_min_height = 3;

	// 连续多帧确认，降低单帧误检概率。
	constexpr int confirm_frames = 3;
}

struct ZebraResult {
	bool detected = false;
	int block_count = 0;
	int max_area = 0;
	vector<Rect> blocks;
};

static vector<Point> createRoiVertices(int rows, int cols) {
	return {
		Point(0, static_cast<int>(rows * ZebraConfig::roi_top_ratio)),
		Point(cols - 1, static_cast<int>(rows * ZebraConfig::roi_top_ratio)),
		Point(cols - 1, rows - 1),
		Point(0, rows - 1)
	};
}

static Mat createWhiteRoiMask(const Mat& frame, double& used_threshold) {
	Mat resized;
	resize(
		frame,
		resized,
		Size(ZebraConfig::image_width, ZebraConfig::image_height)
	);

	Mat gray;
	cvtColor(resized, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0);

	const int rows = gray.rows;
	const int cols = gray.cols;
	const int start_y = static_cast<int>(rows * ZebraConfig::roi_top_ratio);

	// 只使用道路候选区域计算 OTSU，避免上半幅天空和背景干扰阈值。
	Mat road_gray = gray(Rect(0, start_y, cols, rows - start_y));
	Mat road_binary;
	double otsu_threshold = threshold(
		road_gray,
		road_binary,
		0,
		255,
		THRESH_BINARY | THRESH_OTSU
	);
	used_threshold = max(otsu_threshold, ZebraConfig::otsu_threshold_min);
	if (used_threshold != otsu_threshold) {
		threshold(road_gray, road_binary, used_threshold, 255, THRESH_BINARY);
	}

	Mat white_mask = Mat::zeros(gray.size(), CV_8UC1);
	road_binary.copyTo(white_mask(Rect(0, start_y, cols, rows - start_y)));

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(white_mask, white_mask, MORPH_OPEN, kernel);
	morphologyEx(white_mask, white_mask, MORPH_CLOSE, kernel);

	vector<Point> vertices = createRoiVertices(rows, cols);

	Mat polygon_mask = Mat::zeros(white_mask.size(), CV_8UC1);
	vector<vector<Point>> polygons = {vertices};
	fillPoly(polygon_mask, polygons, Scalar(255));

	Mat roi_mask;
	bitwise_and(white_mask, polygon_mask, roi_mask);
	return roi_mask;
}

static ZebraResult detectZebraCrossing(const Mat& roi_mask) {
	ZebraResult result;
	vector<vector<Point>> contours;
	findContours(roi_mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		double area = contourArea(contour);
		Rect box = boundingRect(contour);

		if (area >= ZebraConfig::block_min_area &&
			box.width >= ZebraConfig::block_min_width &&
			box.height >= ZebraConfig::block_min_height) {
			result.block_count++;
			result.max_area = max(result.max_area, static_cast<int>(area));
			result.blocks.push_back(box);
		}
	}

	sort(result.blocks.begin(), result.blocks.end(), [](const Rect& a, const Rect& b) {
		return (a.x + a.width / 2) < (b.x + b.width / 2);
	});

	result.detected = result.block_count >= ZebraConfig::required_blocks;

	return result;
}

int main() {
	VideoCapture camera(0, CAP_V4L2);
	if (!camera.isOpened()) {
		cerr << "Camera open failed" << endl;
		return 1;
	}

	camera.set(CAP_PROP_FRAME_WIDTH, ZebraConfig::capture_width);
	camera.set(CAP_PROP_FRAME_HEIGHT, ZebraConfig::capture_height);
	camera.set(CAP_PROP_FPS, 30);
	camera.set(CAP_PROP_BUFFERSIZE, 1);

	int detection_count = 0;
	bool zebra_confirmed = false;
	int frame_count = 0;

	namedWindow("zebra_detection", WINDOW_NORMAL);
	resizeWindow("zebra_detection", 1280, 480);

	cout << "Zebra crossing visual test started." << endl;
	cout << "Press ESC, q or Q to stop." << endl;

	while (true) {
		Mat frame;
		if (!camera.grab() || !camera.retrieve(frame) || frame.empty()) {
			cerr << "Camera frame read failed" << endl;
			break;
		}

		double otsu_threshold = 0.0;
		Mat roi_mask = createWhiteRoiMask(frame, otsu_threshold);
		ZebraResult result = detectZebraCrossing(roi_mask);
		frame_count++;

		if (result.detected) {
			detection_count++;
		} else {
			detection_count = 0;
		}

		if (!zebra_confirmed &&
			detection_count >= ZebraConfig::confirm_frames) {
			zebra_confirmed = true;
			cout << "ZEBRA_CROSSING_DETECTED" << endl;
		}

		// 离开斑马线后允许下一次识别。
		if (zebra_confirmed && detection_count == 0) {
			zebra_confirmed = false;
			cout << "ZEBRA_CROSSING_CLEARED" << endl;
		}

		if (frame_count % 10 == 0) {
			cout << "DETECTED=" << result.detected
				 << " OTSU=" << static_cast<int>(otsu_threshold)
				 << " BLOCKS=" << result.block_count
				 << " MAX_AREA=" << result.max_area
				 << " CONFIRM_COUNT=" << detection_count
				 << endl;
		}

		Mat display;
		resize(
			frame,
			display,
			Size(ZebraConfig::image_width, ZebraConfig::image_height)
		);
		vector<vector<Point>> roi_outline = {
			createRoiVertices(display.rows, display.cols)
		};
		polylines(display, roi_outline, true, Scalar(0, 255, 255), 2);
		for (size_t i = 0; i < result.blocks.size(); ++i) {
			const Rect& block = result.blocks[i];
			rectangle(display, block, Scalar(0, 0, 255), 2);
			putText(
				display,
				to_string(i + 1),
				Point(block.x, max(20, block.y - 5)),
				FONT_HERSHEY_SIMPLEX,
				0.6,
				Scalar(0, 0, 255),
				2,
				LINE_AA
			);
		}

		const bool confirmed = zebra_confirmed;
		putText(
			display,
			confirmed ? "ZEBRA DETECTED" : "SEARCHING",
			Point(15, 35),
			FONT_HERSHEY_SIMPLEX,
			0.9,
			confirmed ? Scalar(0, 0, 255) : Scalar(0, 255, 0),
			2,
			LINE_AA
		);

		Mat mask_bgr;
		cvtColor(roi_mask, mask_bgr, COLOR_GRAY2BGR);
		Mat combined;
		hconcat(display, mask_bgr, combined);
		imshow("zebra_detection", combined);

		int key = waitKey(1);
		if (key == 27 || key == 'q' || key == 'Q') {
			break;
		}
	}

	camera.release();
	destroyAllWindows();
	return 0;
}
