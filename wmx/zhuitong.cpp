// 蓝色锥桶纯视觉识别测试
// 比赛锥桶尺寸：高度约 8 cm，底部直径约 7.8 cm。
// 程序按颜色、轮廓面积和外接矩形形状过滤，并在画面中框出候选锥桶。

#include <arpa/inet.h>
#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;

namespace ConeConfig {
	constexpr int image_width = 640;
	constexpr int image_height = 480;
	constexpr int camera_index = 0;
	constexpr int http_port = 8090;
	constexpr int stream_width = 320;
	constexpr int stream_height = 240;
	constexpr int jpeg_quality = 45;
	constexpr int stream_fps = 20;
	constexpr const char* pi_ip = "192.168.137.161";
	constexpr size_t max_cones = 2;

	// 锥桶在地面上，主要检测画面下方 35%～100%。
	constexpr double roi_top_ratio = 0.35;

	// 参考现有工程的蓝色 HSV 阈值，并适当放宽以识别远处小目标。
	constexpr int blue_h_min = 95;
	constexpr int blue_h_max = 145;
	constexpr int blue_s_min = 55;
	constexpr int blue_v_min = 45;

	// 640×480 下的初始过滤参数，需要根据现场距离调节。
	constexpr double min_area = 25.0;
	constexpr double max_area = 50000.0;
	constexpr int min_width = 4;
	constexpr int min_height = 4;
	constexpr int max_width = 300;
	constexpr int max_height = 300;

	// 8cm 高、7.8cm 底径接近 1:1；考虑透视、遮挡和检测误差，范围放宽。
	constexpr double min_aspect_ratio = 0.35; // width / height
	constexpr double max_aspect_ratio = 1.80;
	constexpr double min_fill_ratio = 0.20;
}

struct ConeDetection {
	Rect box;
	double area = 0.0;
	Point center;
};

enum class StreamMode {
	Raw,
	Processed,
	Both
};

static atomic<bool> g_running(true);
static mutex g_frame_mutex;
static Mat g_latest_raw;
static Mat g_latest_processed;

static void onSignal(int) {
	g_running = false;
}

static bool sendAll(int fd, const void* data, size_t len) {
	const char* p = static_cast<const char*>(data);
	while (len > 0) {
		ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
		if (n <= 0) {
			return false;
		}
		p += n;
		len -= static_cast<size_t>(n);
	}
	return true;
}

static bool sendString(int fd, const string& s) {
	return sendAll(fd, s.data(), s.size());
}

static string parsePath(const char* request_buf) {
	string req(request_buf ? request_buf : "");
	size_t first_space = req.find(' ');
	if (first_space == string::npos) return "/";
	size_t second_space = req.find(' ', first_space + 1);
	if (second_space == string::npos) return "/";
	return req.substr(first_space + 1, second_space - first_space - 1);
}

static StreamMode parseStreamMode(const string& path) {
	if (path.find("mode=raw") != string::npos) return StreamMode::Raw;
	if (path.find("mode=both") != string::npos) return StreamMode::Both;
	return StreamMode::Processed;
}

static string makeHtmlPage() {
	return R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>蓝色锥桶识别图传</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; background: #020617; color: #e5e7eb; font-family: Arial, "Microsoft YaHei", sans-serif; overflow: hidden; }
.bar { height: 58px; display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: #0f172a; border-bottom: 1px solid rgba(148,163,184,.25); }
.title { font-size: 17px; font-weight: 700; margin-right: auto; white-space: nowrap; }
button { border: 0; border-radius: 10px; padding: 9px 12px; color: white; background: #2563eb; font-size: 14px; cursor: pointer; }
button.active { background: #16a34a; }
button.recording { background: #dc2626; }
#stage { width: 100vw; height: calc(100vh - 58px); display: flex; align-items: center; justify-content: center; background: #000; }
#stream { width: 100%; height: 100%; object-fit: contain; background: #000; }
#tip { position: fixed; left: 10px; bottom: 10px; padding: 7px 9px; border-radius: 8px; background: rgba(15,23,42,.72); color: #cbd5e1; font-size: 13px; }
@media (max-width: 620px) { .title { display: none; } button { padding: 8px 9px; font-size: 13px; } }
</style>
</head>
<body>
<div class="bar">
  <div class="title">蓝色锥桶识别图传</div>
  <button id="rawBtn">原图</button>
  <button id="processedBtn" class="active">处理图</button>
  <button id="bothBtn">同屏双图</button>
  <button id="shotBtn">截图</button>
  <button id="recordBtn">开始录屏</button>
  <button id="fullBtn">全屏</button>
</div>
<div id="stage"><img id="stream" src="/stream?mode=processed" alt="stream"></div>
<div id="tip">可切换原图 / 处理图 / 同屏双图；截图 PNG；录屏 WEBM。</div>
<canvas id="canvas" style="display:none"></canvas>
<script>
const img = document.getElementById('stream');
const canvas = document.getElementById('canvas');
const rawBtn = document.getElementById('rawBtn');
const processedBtn = document.getElementById('processedBtn');
const bothBtn = document.getElementById('bothBtn');
const shotBtn = document.getElementById('shotBtn');
const recordBtn = document.getElementById('recordBtn');
const fullBtn = document.getElementById('fullBtn');
let recorder = null, chunks = [], drawTimer = null;

function setMode(mode) {
	img.onload = null;
	img.onerror = null;
	img.removeAttribute('src');
  img.src = '/stream?mode=' + mode + '&t=' + Date.now();
  rawBtn.classList.toggle('active', mode === 'raw');
  processedBtn.classList.toggle('active', mode === 'processed');
  bothBtn.classList.toggle('active', mode === 'both');
}
rawBtn.onclick = () => setMode('raw');
processedBtn.onclick = () => setMode('processed');
bothBtn.onclick = () => setMode('both');

function timeName(prefix, ext) {
  const d = new Date(); const pad = n => String(n).padStart(2, '0');
  return `${prefix}_${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.${ext}`;
}
function drawFrame() {
  const w = img.naturalWidth || img.clientWidth || 640;
  const h = img.naturalHeight || img.clientHeight || 480;
  canvas.width = w; canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0, w, h);
}
function downloadBlob(blob, name) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
shotBtn.onclick = () => { drawFrame(); canvas.toBlob(b => { if (b) downloadBlob(b, timeName('cone_shot', 'png')); }, 'image/png'); };
recordBtn.onclick = () => {
  if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
  drawFrame(); chunks = [];
  recorder = new MediaRecorder(canvas.captureStream(20), { mimeType: 'video/webm' });
  recorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = () => { clearInterval(drawTimer); drawTimer = null; recordBtn.textContent = '开始录屏'; recordBtn.classList.remove('recording'); downloadBlob(new Blob(chunks, {type:'video/webm'}), timeName('cone_record', 'webm')); };
  drawTimer = setInterval(drawFrame, 50); recorder.start(); recordBtn.textContent = '停止录屏'; recordBtn.classList.add('recording');
};
fullBtn.onclick = () => { if (!document.fullscreenElement) document.documentElement.requestFullscreen?.(); else document.exitFullscreen?.(); };
</script>
</body>
</html>)HTML";
}

static Mat makeStreamFrame(StreamMode mode) {
	Mat raw;
	Mat processed;
	{
		lock_guard<mutex> lock(g_frame_mutex);
		if (!g_latest_raw.empty()) raw = g_latest_raw.clone();
		if (!g_latest_processed.empty()) processed = g_latest_processed.clone();
	}

	if (raw.empty() && processed.empty()) {
		Mat waiting(ConeConfig::image_height, ConeConfig::image_width, CV_8UC3, Scalar(20, 20, 20));
		putText(waiting, "Waiting for camera frame...", Point(30, ConeConfig::image_height / 2), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, LINE_AA);
		return waiting;
	}
	if (raw.empty()) raw = processed.clone();
	if (processed.empty()) processed = raw.clone();

	if (mode == StreamMode::Raw) {
		return raw;
	}
	if (mode == StreamMode::Both) {
		Mat raw_show;
		Mat processed_show;
		resize(raw, raw_show, Size(ConeConfig::stream_width, ConeConfig::stream_height));
		resize(processed, processed_show, Size(ConeConfig::stream_width, ConeConfig::stream_height));
		Mat both;
		hconcat(raw_show, processed_show, both);
		return both;
	}
	return processed;
}

static void handleStreamClient(int client_fd, StreamMode mode) {
	string header =
		"HTTP/1.0 200 OK\r\n"
		"Server: cone-mjpeg\r\n"
		"Connection: close\r\n"
		"Max-Age: 0\r\n"
		"Expires: 0\r\n"
		"Cache-Control: no-cache, private\r\n"
		"Pragma: no-cache\r\n"
		"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
	if (!sendString(client_fd, header)) return;

	vector<int> jpeg_params = {IMWRITE_JPEG_QUALITY, ConeConfig::jpeg_quality};
	vector<uchar> jpg;
	const int delay_ms = 1000 / ConeConfig::stream_fps;
	while (g_running) {
		Mat out = makeStreamFrame(mode);
		if (out.empty()) continue;
		if (!imencode(".jpg", out, jpg, jpeg_params)) continue;

		string part_header =
			"--frame\r\n"
			"Content-Type: image/jpeg\r\n"
			"Content-Length: " + to_string(jpg.size()) + "\r\n\r\n";
		if (!sendString(client_fd, part_header)) break;
		if (!sendAll(client_fd, jpg.data(), jpg.size())) break;
		if (!sendString(client_fd, "\r\n")) break;
		this_thread::sleep_for(chrono::milliseconds(delay_ms));
	}
}

static void httpServerThread() {
	int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
	if (server_fd < 0) {
		cerr << "HTTP socket create failed" << endl;
		return;
	}

	int opt = 1;
	::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

	sockaddr_in server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	server_addr.sin_port = htons(static_cast<uint16_t>(ConeConfig::http_port));

	if (::bind(server_fd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
		cerr << "HTTP bind failed, port=" << ConeConfig::http_port << endl;
		::close(server_fd);
		return;
	}
	if (::listen(server_fd, 2) < 0) {
		cerr << "HTTP listen failed" << endl;
		::close(server_fd);
		return;
	}

	string url = "http://" + string(ConeConfig::pi_ip) + ":" + to_string(ConeConfig::http_port) + "/";
	cout << endl;
	cout << "========================================" << endl;
	cout << "锥桶识别图传网页地址：" << url << endl;
	cout << "MobaXterm 中可按 Ctrl 后点击，或直接选中复制" << endl;
	cout << "Clickable: \033]8;;" << url << "\033\\" << url << "\033]8;;\033\\" << endl;
	cout << "========================================" << endl;
	cout << endl;

	while (g_running) {
		sockaddr_in client_addr;
		socklen_t client_len = sizeof(client_addr);
		int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
		if (client_fd < 0) {
			if (g_running) cerr << "HTTP accept failed" << endl;
			continue;
		}

		char request_buf[1024] = {0};
		::recv(client_fd, request_buf, sizeof(request_buf) - 1, 0);
		string path = parsePath(request_buf);

		if (path.find("/stream") == 0) {
			StreamMode mode = parseStreamMode(path);
			thread([client_fd, mode]() {
				handleStreamClient(client_fd, mode);
				::close(client_fd);
			}).detach();
			continue;
		} else {
			string body = makeHtmlPage();
			string header =
				"HTTP/1.0 200 OK\r\n"
				"Server: cone-mjpeg\r\n"
				"Connection: close\r\n"
				"Cache-Control: no-cache\r\n"
				"Content-Type: text/html; charset=utf-8\r\n"
				"Content-Length: " + to_string(body.size()) + "\r\n\r\n";
			sendString(client_fd, header);
			sendString(client_fd, body);
		}

		::close(client_fd);
	}

	::close(server_fd);
}

static Mat createBlueMask(const Mat& resized) {
	Mat hsv;
	cvtColor(resized, hsv, COLOR_BGR2HSV);

	const int roi_top = static_cast<int>(
		ConeConfig::image_height * ConeConfig::roi_top_ratio
	);
	Rect roi_rect(
		0,
		roi_top,
		ConeConfig::image_width,
		ConeConfig::image_height - roi_top
	);
	Mat hsv_roi = hsv(roi_rect);

	Mat roi_mask;
	inRange(
		hsv_roi,
		Scalar(
			ConeConfig::blue_h_min,
			ConeConfig::blue_s_min,
			ConeConfig::blue_v_min
		),
		Scalar(ConeConfig::blue_h_max, 255, 255),
		roi_mask
	);

	// 小核开运算去噪，闭运算连接锥桶内部的小孔洞。
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(roi_mask, roi_mask, MORPH_OPEN, kernel);
	morphologyEx(roi_mask, roi_mask, MORPH_CLOSE, kernel);

	Mat full_mask = Mat::zeros(resized.size(), CV_8UC1);
	roi_mask.copyTo(full_mask(roi_rect));
	return full_mask;
}

static vector<ConeDetection> detectBlueCones(const Mat& blue_mask) {
	vector<vector<Point>> contours;
	findContours(
		blue_mask.clone(),
		contours,
		RETR_EXTERNAL,
		CHAIN_APPROX_SIMPLE
	);

	vector<ConeDetection> detections;
	for (const auto& contour : contours) {
		double area = contourArea(contour);
		if (area < ConeConfig::min_area || area > ConeConfig::max_area) {
			continue;
		}

		Rect box = boundingRect(contour);
		if (box.width < ConeConfig::min_width ||
			box.height < ConeConfig::min_height ||
			box.width > ConeConfig::max_width ||
			box.height > ConeConfig::max_height) {
			continue;
		}

		double aspect = static_cast<double>(box.width) / box.height;
		double fill_ratio = area / static_cast<double>(box.area());
		if (aspect < ConeConfig::min_aspect_ratio ||
			aspect > ConeConfig::max_aspect_ratio ||
			fill_ratio < ConeConfig::min_fill_ratio) {
			continue;
		}

		Moments moments = cv::moments(contour);
		Point center(
			box.x + box.width / 2,
			box.y + box.height / 2
		);
		if (moments.m00 != 0.0) {
			center.x = static_cast<int>(moments.m10 / moments.m00);
			center.y = static_cast<int>(moments.m01 / moments.m00);
		}

		detections.push_back({box, area, center});
	}

	// 优先显示面积更大的近处锥桶。
	sort(
		detections.begin(),
		detections.end(),
		[](const ConeDetection& a, const ConeDetection& b) {
			return a.area > b.area;
		}
	);
	if (detections.size() > ConeConfig::max_cones) {
		detections.resize(ConeConfig::max_cones);
	}
	return detections;
}

int main() {
	signal(SIGINT, onSignal);
	signal(SIGTERM, onSignal);

	VideoCapture camera(ConeConfig::camera_index, CAP_V4L2);
	if (!camera.isOpened()) {
		cerr << "Camera open failed" << endl;
		return 1;
	}

	camera.set(CAP_PROP_FRAME_WIDTH, ConeConfig::image_width);
	camera.set(CAP_PROP_FRAME_HEIGHT, ConeConfig::image_height);
	camera.set(CAP_PROP_FPS, 30);
	camera.set(CAP_PROP_BUFFERSIZE, 1);

	int frame_count = 0;
	cout << "Blue cone visual test started." << endl;
	cout << "HTTP stream only. Press Ctrl+C to stop." << endl;
	thread http_thread(httpServerThread);

	while (g_running) {
		Mat frame;
		if (!camera.grab() || !camera.retrieve(frame) || frame.empty()) {
			cerr << "Camera frame read failed" << endl;
			break;
		}

		Mat display;
		resize(
			frame,
			display,
			Size(ConeConfig::image_width, ConeConfig::image_height)
		);
		Mat raw_display = display.clone();

		Mat blue_mask = createBlueMask(display);
		vector<ConeDetection> cones = detectBlueCones(blue_mask);
		Mat processed_display;
		cvtColor(blue_mask, processed_display, COLOR_GRAY2BGR);
		frame_count++;

		const int roi_top = static_cast<int>(
			ConeConfig::image_height * ConeConfig::roi_top_ratio
		);
		line(
			display,
			Point(0, roi_top),
			Point(ConeConfig::image_width - 1, roi_top),
			Scalar(0, 255, 255),
			2
		);
		line(
			processed_display,
			Point(0, roi_top),
			Point(ConeConfig::image_width - 1, roi_top),
			Scalar(0, 255, 255),
			2
		);

		for (size_t i = 0; i < cones.size(); ++i) {
			const ConeDetection& cone = cones[i];
			rectangle(display, cone.box, Scalar(0, 255, 0), 2);
			rectangle(processed_display, cone.box, Scalar(0, 255, 0), 2);
			circle(display, cone.center, 3, Scalar(0, 0, 255), FILLED);
			circle(processed_display, cone.center, 3, Scalar(0, 0, 255), FILLED);
			putText(
				display,
				string("CONE ") + to_string(i + 1) +
				" A=" + to_string(static_cast<int>(cone.area)),
				Point(cone.box.x, max(20, cone.box.y - 7)),
				FONT_HERSHEY_SIMPLEX,
				0.5,
				Scalar(0, 255, 0),
				1,
				LINE_AA
			);
			putText(
				processed_display,
				string("CONE ") + to_string(i + 1) +
				" A=" + to_string(static_cast<int>(cone.area)),
				Point(cone.box.x, max(20, cone.box.y - 7)),
				FONT_HERSHEY_SIMPLEX,
				0.5,
				Scalar(0, 255, 0),
				1,
				LINE_AA
			);
		}

		putText(
			display,
			string("BLUE CONES=") + to_string(cones.size()),
			Point(15, 35),
			FONT_HERSHEY_SIMPLEX,
			0.8,
			cones.empty() ? Scalar(0, 0, 255) : Scalar(0, 255, 0),
			2,
			LINE_AA
		);
		putText(
			processed_display,
			string("BLUE CONES=") + to_string(cones.size()),
			Point(15, 35),
			FONT_HERSHEY_SIMPLEX,
			0.8,
			cones.empty() ? Scalar(0, 0, 255) : Scalar(0, 255, 0),
			2,
			LINE_AA
		);

		{
			lock_guard<mutex> lock(g_frame_mutex);
			g_latest_raw = raw_display.clone();
			g_latest_processed = processed_display.clone();
		}

		if (frame_count % 10 == 0) {
			cout << "BLUE_CONES=" << cones.size();
			if (!cones.empty()) {
				cout << " PRIMARY_X=" << cones.front().center.x
					 << " PRIMARY_Y=" << cones.front().center.y
					 << " PRIMARY_AREA=" << static_cast<int>(cones.front().area);
			}
			cout << endl;
		}

		this_thread::sleep_for(chrono::milliseconds(1));
	}

	if (http_thread.joinable()) {
		http_thread.detach();
	}
	camera.release();
	destroyAllWindows();
	return 0;
}
