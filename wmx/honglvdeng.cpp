// 红绿灯纯视觉识别测试
// 工作流程：摄像头 -> HSV 分割红/绿灯 -> 轮廓过滤 -> 连续多帧确认。
// 本程序不控制 GPIO、电机或舵机，只显示并打印识别结果。

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

namespace TrafficConfig {
	constexpr int image_width = 640;
	constexpr int image_height = 480;
	constexpr int camera_index = 0;

	// 红绿灯安装在赛道上方，优先检测画面上部 5%～70%。
	constexpr double roi_top_ratio = 0.05;
	constexpr double roi_bottom_ratio = 0.70;

	// OpenCV HSV：H 范围为 0～179。红色跨越 H 的首尾两段。
	constexpr int red1_h_min = 0;
	constexpr int red1_h_max = 12;
	constexpr int red2_h_min = 165;
	constexpr int red2_h_max = 179;
	constexpr int red_s_min = 100;
	constexpr int red_v_min = 100;

	constexpr int green_h_min = 35;
	constexpr int green_h_max = 95;
	constexpr int green_s_min = 80;
	constexpr int green_v_min = 80;

	// 远处灯面积较小，首次测试使用较低阈值。
	constexpr double min_area = 20.0;
	constexpr double max_area = 12000.0;
	constexpr int min_width = 4;
	constexpr int min_height = 4;
	constexpr double min_aspect_ratio = 0.45;
	constexpr double max_aspect_ratio = 2.20;
	constexpr double min_fill_ratio = 0.35;

	constexpr int confirm_frames = 3;
	constexpr int clear_frames = 5;
}

enum class LightState {
	Unknown,
	Red,
	Green
};

struct ColorDetection {
	bool found = false;
	double area = 0.0;
	Rect box;
};

struct TrafficResult {
	ColorDetection red;
	ColorDetection green;
	Mat red_mask;
	Mat green_mask;
};

static const char* stateName(LightState state) {
	switch (state) {
	case LightState::Red:
		return "RED";
	case LightState::Green:
		return "GREEN";
	case LightState::Unknown:
	default:
		return "UNKNOWN";
	}
}

static ColorDetection findBestLight(const Mat& mask, int roi_top) {
	ColorDetection best;
	vector<vector<Point>> contours;
	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		double area = contourArea(contour);
		if (area < TrafficConfig::min_area || area > TrafficConfig::max_area) {
			continue;
		}

		Rect box = boundingRect(contour);
		if (box.width < TrafficConfig::min_width ||
			box.height < TrafficConfig::min_height) {
			continue;
		}

		double aspect = static_cast<double>(box.width) / box.height;
		double fill_ratio = area / static_cast<double>(box.area());
		if (aspect < TrafficConfig::min_aspect_ratio ||
			aspect > TrafficConfig::max_aspect_ratio ||
			fill_ratio < TrafficConfig::min_fill_ratio) {
			continue;
		}

		if (!best.found || area > best.area) {
			best.found = true;
			best.area = area;
			best.box = box;
			best.box.y += roi_top;
		}
	}

	return best;
}

static TrafficResult detectTrafficLight(const Mat& frame) {
	TrafficResult result;
	Mat resized;
	resize(
		frame,
		resized,
		Size(TrafficConfig::image_width, TrafficConfig::image_height)
	);

	Mat hsv;
	cvtColor(resized, hsv, COLOR_BGR2HSV);

	const int roi_top = static_cast<int>(
		TrafficConfig::image_height * TrafficConfig::roi_top_ratio
	);
	const int roi_bottom = static_cast<int>(
		TrafficConfig::image_height * TrafficConfig::roi_bottom_ratio
	);
	Rect roi_rect(0, roi_top, TrafficConfig::image_width, roi_bottom - roi_top);
	Mat hsv_roi = hsv(roi_rect);

	Mat red_mask1;
	Mat red_mask2;
	inRange(
		hsv_roi,
		Scalar(TrafficConfig::red1_h_min, TrafficConfig::red_s_min, TrafficConfig::red_v_min),
		Scalar(TrafficConfig::red1_h_max, 255, 255),
		red_mask1
	);
	inRange(
		hsv_roi,
		Scalar(TrafficConfig::red2_h_min, TrafficConfig::red_s_min, TrafficConfig::red_v_min),
		Scalar(TrafficConfig::red2_h_max, 255, 255),
		red_mask2
	);
	bitwise_or(red_mask1, red_mask2, result.red_mask);

	inRange(
		hsv_roi,
		Scalar(TrafficConfig::green_h_min, TrafficConfig::green_s_min, TrafficConfig::green_v_min),
		Scalar(TrafficConfig::green_h_max, 255, 255),
		result.green_mask
	);

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(result.red_mask, result.red_mask, MORPH_OPEN, kernel);
	morphologyEx(result.red_mask, result.red_mask, MORPH_CLOSE, kernel);
	morphologyEx(result.green_mask, result.green_mask, MORPH_OPEN, kernel);
	morphologyEx(result.green_mask, result.green_mask, MORPH_CLOSE, kernel);

	result.red = findBestLight(result.red_mask, roi_top);
	result.green = findBestLight(result.green_mask, roi_top);
	return result;
}

static LightState chooseCandidate(const TrafficResult& result) {
	if (result.red.found && result.green.found) {
		// 两种颜色同时出现时，选择有效轮廓面积更大的灯。
		return result.red.area >= result.green.area
			? LightState::Red
			: LightState::Green;
	}
	if (result.red.found) {
		return LightState::Red;
	}
	if (result.green.found) {
		return LightState::Green;
	}
	return LightState::Unknown;
}

int main() {
	VideoCapture camera(TrafficConfig::camera_index, CAP_V4L2);
	if (!camera.isOpened()) {
		cerr << "Camera open failed" << endl;
		return 1;
	}

	camera.set(CAP_PROP_FRAME_WIDTH, TrafficConfig::image_width);
	camera.set(CAP_PROP_FRAME_HEIGHT, TrafficConfig::image_height);
	camera.set(CAP_PROP_FPS, 30);
	camera.set(CAP_PROP_BUFFERSIZE, 1);

	namedWindow("traffic_light", WINDOW_NORMAL);
	resizeWindow("traffic_light", 960, 720);

	LightState stable_state = LightState::Unknown;
	LightState candidate_state = LightState::Unknown;
	int candidate_count = 0;
	int unknown_count = 0;
	int frame_count = 0;

	cout << "Traffic light visual test started." << endl;
	cout << "Press ESC, q or Q to stop." << endl;

	while (true) {
		Mat frame;
		if (!camera.grab() || !camera.retrieve(frame) || frame.empty()) {
			cerr << "Camera frame read failed" << endl;
			break;
		}

		TrafficResult result = detectTrafficLight(frame);
		LightState current_candidate = chooseCandidate(result);
		frame_count++;

		if (current_candidate == LightState::Unknown) {
			unknown_count++;
			candidate_count = 0;
			candidate_state = LightState::Unknown;
			if (unknown_count >= TrafficConfig::clear_frames) {
				stable_state = LightState::Unknown;
			}
		} else {
			unknown_count = 0;
			if (current_candidate == candidate_state) {
				candidate_count++;
			} else {
				candidate_state = current_candidate;
				candidate_count = 1;
			}

			if (candidate_count >= TrafficConfig::confirm_frames &&
				stable_state != candidate_state) {
				stable_state = candidate_state;
				cout << "TRAFFIC_LIGHT=" << stateName(stable_state) << endl;
			}
		}

		Mat display;
		resize(
			frame,
			display,
			Size(TrafficConfig::image_width, TrafficConfig::image_height)
		);

		const int roi_top = static_cast<int>(
			TrafficConfig::image_height * TrafficConfig::roi_top_ratio
		);
		const int roi_bottom = static_cast<int>(
			TrafficConfig::image_height * TrafficConfig::roi_bottom_ratio
		);
		rectangle(
			display,
			Rect(0, roi_top, TrafficConfig::image_width, roi_bottom - roi_top),
			Scalar(255, 255, 0),
			2
		);

		if (result.red.found) {
			rectangle(display, result.red.box, Scalar(0, 0, 255), 2);
		}
		if (result.green.found) {
			rectangle(display, result.green.box, Scalar(0, 255, 0), 2);
		}

		Scalar state_color = Scalar(0, 255, 255);
		if (stable_state == LightState::Red) {
			state_color = Scalar(0, 0, 255);
		} else if (stable_state == LightState::Green) {
			state_color = Scalar(0, 255, 0);
		}
		putText(
			display,
			string("STATE=") + stateName(stable_state),
			Point(15, 35),
			FONT_HERSHEY_SIMPLEX,
			0.9,
			state_color,
			2,
			LINE_AA
		);

		if (frame_count % 10 == 0) {
			cout << "STATE=" << stateName(stable_state)
				 << " RED_FOUND=" << result.red.found
				 << " RED_AREA=" << static_cast<int>(result.red.area)
				 << " GREEN_FOUND=" << result.green.found
				 << " GREEN_AREA=" << static_cast<int>(result.green.area)
				 << endl;
		}

		imshow("traffic_light", display);
		int key = waitKey(1);
		if (key == 27 || key == 'q' || key == 'Q') {
			break;
		}
	}

	camera.release();
	destroyAllWindows();
	return 0;
}
