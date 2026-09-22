// 蓝色挡板移开启动测试程序
//
// 比赛要求对应流程：
// 1. 自动驾驶程序启动后，先识别程序切换线前的蓝色挡板；
// 2. 确认挡板存在后，等待裁判将挡板移开；
// 3. 连续多帧确认蓝色区域消失；
// 4. 延时后启动电机，进入基础运行测试。
//
// 更新说明：摄像头改为先采集 1920x1080，再缩放到 320x180 进行挡板检测；保留原有多帧确认逻辑。
// 更新日期：2026-09

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;

namespace DangBanConfig {
	constexpr int capture_width = 1920;
	constexpr int capture_height = 1080;
	constexpr int image_width = 320;
	constexpr int image_height = 180;

	// 参考 g5g-new/bule_card.cpp 的蓝色 HSV 范围。
	constexpr int blue_h_min = 100;
	constexpr int blue_h_max = 140;
	constexpr int blue_s_min = 50;
	constexpr int blue_v_min = 0;

	// 挡板位于摄像头画面中部时使用该 ROI。
	// 如实际摄像头画面中挡板位置不同，可改为 0 和 image_height。
	constexpr int roi_top = 40;
	constexpr int roi_bottom = 140;

	// 蓝色面积阈值：需要根据实际画面调试。
	// 当前 area 是 ROI 内有效蓝色轮廓的 contourArea，不是整幅图像像素数。
	// 当前画面为 640*360，ROI 高度已按比例调整。
	constexpr double board_present_area = 15000.0;
	constexpr double board_removed_area = 7500.0;

	// 连续帧确认，避免单帧误识别导致车辆启动。
	constexpr int present_confirm_frames = 15;
	constexpr int removed_confirm_frames = 15;

	// 挡板移开确认后，给裁判和车辆一个短暂缓冲时间。
	constexpr int start_delay_ms = 2000;

}

enum class StartState {
	WaitingForBoard,
	BoardDetected,
	WaitingForBoardRemoved,
	ReadyToStart,
	Running
};

struct BlueDetection {
	double area = 0.0;
	Mat mask;
};

static BlueDetection detectBlueBoard(const Mat& frame) {
	BlueDetection result;

	if (frame.empty()) {
		return result;
	}

	Mat resized;
	resize(
		frame,
		resized,
		Size(DangBanConfig::image_width, DangBanConfig::image_height)
	);

	Mat hsv;
	cvtColor(resized, hsv, COLOR_BGR2HSV);

	Mat full_mask;
	inRange(
		hsv,
		Scalar(
			DangBanConfig::blue_h_min,
			DangBanConfig::blue_s_min,
			DangBanConfig::blue_v_min
		),
		Scalar(DangBanConfig::blue_h_max, 255, 255),
		full_mask
	);

	int roi_height = DangBanConfig::roi_bottom - DangBanConfig::roi_top;
	Rect roi_rect(
		0,
		DangBanConfig::roi_top,
		DangBanConfig::image_width,
		roi_height
	);
	result.mask = full_mask(roi_rect).clone();

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(result.mask, result.mask, MORPH_OPEN, kernel);
	morphologyEx(result.mask, result.mask, MORPH_CLOSE, kernel);

	vector<vector<Point>> contours;
	findContours(
		result.mask,
		contours,
		RETR_EXTERNAL,
		CHAIN_APPROX_SIMPLE
	);

	for (const auto& contour : contours) {
		double contour_area = contourArea(contour);
		result.area += contour_area;
	}

	return result;
}

// OpenCV 自带 Hershey 字体不支持中文；状态信息统一打印到终端。
static const char* stateName(StartState state) {
	switch (state) {
	case StartState::WaitingForBoard:
		return "WAIT_BOARD";
	case StartState::BoardDetected:
		return "BOARD_FOUND";
	case StartState::WaitingForBoardRemoved:
		return "WAIT_REMOVED";
	case StartState::ReadyToStart:
		return "READY_START";
	case StartState::Running:
		return "RUNNING";
	}
	return "UNKNOWN";
}

int main() {
	VideoCapture camera(-1);
	if (!camera.isOpened()) {
		cerr << "摄像头打开失败" << endl;
		return 1;
	}

	camera.set(CAP_PROP_FRAME_WIDTH, DangBanConfig::capture_width);
	camera.set(CAP_PROP_FRAME_HEIGHT, DangBanConfig::capture_height);
	// 尽量只保留一帧，减少 USB 摄像头缓冲造成的画面延迟。
	camera.set(CAP_PROP_BUFFERSIZE, 1);

	StartState state = StartState::WaitingForBoard;
	int present_count = 0;
	int removed_count = 0;
	StartState last_printed_state = state;
	int frame_count = 0;

	cout << "蓝色挡板视觉测试开始" << endl;
	cout << "请先将蓝色挡板放在车前，程序会先确认挡板存在。" << endl;
	cout << "移开挡板后，程序只显示视觉状态，不会控制电机。" << endl;
	cout << "按 ESC、q 或 Q 退出。" << endl;

	while (true) {
		Mat frame;
		if (!camera.grab() || !camera.retrieve(frame) || frame.empty()) {
			cerr << "Camera frame read failed" << endl;
			break;
		}

		BlueDetection detection = detectBlueBoard(frame);
		frame_count++;
		bool board_is_present =
			detection.area >= DangBanConfig::board_present_area;
		bool board_is_removed =
			detection.area <= DangBanConfig::board_removed_area;

		switch (state) {
		case StartState::WaitingForBoard:
			if (board_is_present) {
				present_count++;
			} else {
				present_count = 0;
			}

			if (present_count >= DangBanConfig::present_confirm_frames) {
				state = StartState::WaitingForBoardRemoved;
				removed_count = 0;
				cout << "Blue board found; waiting for removal." << endl;
			}
			break;

		case StartState::BoardDetected:
			state = StartState::WaitingForBoardRemoved;
			break;

		case StartState::WaitingForBoardRemoved:
			if (board_is_removed) {
				removed_count++;
			} else {
				removed_count = 0;
			}

			if (removed_count >= DangBanConfig::removed_confirm_frames) {
				state = StartState::ReadyToStart;
				cout << "Board removal confirmed." << endl;
			}
			break;

		case StartState::ReadyToStart:
			this_thread::sleep_for(
				chrono::milliseconds(DangBanConfig::start_delay_ms)
			);
			state = StartState::Running;
			cout << "Visual start condition reached." << endl;
			break;

		case StartState::Running:
			break;
		}

		// 只在状态变化或每 10 帧打印一次，避免终端输出拖慢画面。
		if (state != last_printed_state || frame_count % 10 == 0) {
			cout << "STATE=" << stateName(state)
				 << " BLUE_AREA=" << static_cast<int>(detection.area)
				 << " PRESENT_COUNT=" << present_count
				 << " REMOVED_COUNT=" << removed_count
				 << endl;
			last_printed_state = state;
		}

		this_thread::sleep_for(chrono::milliseconds(1));
	}

	camera.release();

	cout << "Visual test finished." << endl;
	return 0;
}
