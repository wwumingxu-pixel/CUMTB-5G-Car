// Second-camera traffic-light network tester.
// No grayscale/binary panel detection and no GPIO control.
// 更新说明：红绿灯检测改为 1920x1080 采集后缩放处理，并采用“红灯持续确认 -> 绿灯连续帧确认”的状态机逻辑。
// 更新日期：2026-09

#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;

namespace TrafficConfig {
constexpr int capture_width = 1920;
constexpr int capture_height = 1080;
constexpr int process_width = 640;
constexpr int process_height = 360;
constexpr int jpeg_quality = 65;
constexpr int green_confirm_frames = 10;
constexpr int red_confirm_ms = 3000;

// Initial ROI is the upper 75% of the 640x360 processing frame. Set a tighter
// fixed rectangle from the web page after seeing the second-camera stream.
constexpr int default_roi_x = 120;
constexpr int default_roi_y = 80;
constexpr int default_roi_width = 400;
constexpr int default_roi_height = 200;

// OpenCV hue is 0..179. Red wraps around both ends of the hue range.
constexpr int red1_h_min = 0;
constexpr int red1_h_max = 12;
constexpr int red2_h_min = 165;
constexpr int red2_h_max = 179;
constexpr int red_s_min = 140;
constexpr int red_v_min = 140;
constexpr int green_h_min = 35;
constexpr int green_h_max = 95;
constexpr int green_s_min = 110;
constexpr int green_v_min = 130;
constexpr int color_min_area = 50;
constexpr double color_min_fill_ratio = 0.45;
}

enum class TrafficState : int {
    WaitingForRed,
    WaitingForGreen,
    Complete,
};

enum StreamView : size_t {
    kRawView = 0,
    kDebugView = 1,
    kMaskView = 2,
    kStreamViewCount = 3,
};

struct RoiSettings {
    int x = TrafficConfig::default_roi_x;
    int y = TrafficConfig::default_roi_y;
    int width = TrafficConfig::default_roi_width;
    int height = TrafficConfig::default_roi_height;
};

struct ColorDetection {
    bool found = false;
    double area = 0.0;
    Rect box;
    Mat mask;
};

struct DetectionResult {
    Mat raw;
    Mat debug;
    Mat mask_frame;
    Rect roi;
    ColorDetection color;
    TrafficState state = TrafficState::WaitingForRed;
    int confirm_count = 0;
    int red_elapsed_ms = 0;
    bool checking_red = true;
};

struct RawStreamFrames {
    mutex lock;
    condition_variable ready;
    array<Mat, kStreamViewCount> images;
    uint64_t sequence = 0;
};

struct StreamFrames {
    StreamFrames() {
        for (atomic<int>& count : viewers) count.store(0);
    }

    mutex lock;
    condition_variable ready;
    array<vector<uint8_t>, kStreamViewCount> jpeg;
    array<uint64_t, kStreamViewCount> sequence{};
    array<atomic<int>, kStreamViewCount> viewers;
};

static atomic<bool> g_running(true);
static atomic<bool> g_reset_requested(false);
static mutex g_roi_lock;
static RoiSettings g_roi;
static atomic<int> g_traffic_state(static_cast<int>(TrafficState::WaitingForRed));
static atomic<int> g_confirm_count(0);
static atomic<int> g_red_elapsed_ms(0);
static atomic<int> g_red_area(0);
static atomic<int> g_green_area(0);

static const char* stateName(TrafficState state) {
    switch (state) {
    case TrafficState::WaitingForRed: return "WAIT_RED";
    case TrafficState::WaitingForGreen: return "WAIT_GREEN";
    case TrafficState::Complete: return "COMPLETE";
    }
    return "UNKNOWN";
}

static void stopOnSignal(int) {
    g_running = false;
}

static string argument(int argc, char** argv, const string& name,
                       const string& fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (string(argv[index]) == name) return argv[index + 1];
    }
    return fallback;
}

static string fourccName(int fourcc) {
    string result(4, ' ');
    for (int index = 0; index < 4; ++index) {
        result[index] = static_cast<char>((fourcc >> (8 * index)) & 0xff);
    }
    return result;
}

static bool sendAll(int fd, const void* data, size_t size) {
    const char* current = static_cast<const char*>(data);
    while (size > 0) {
        const ssize_t sent = send(fd, current, size, MSG_NOSIGNAL);
        if (sent <= 0) return false;
        current += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

static bool sendText(int fd, const string& text, const string& content_type) {
    const string header = "HTTP/1.0 200 OK\r\nConnection: close\r\n"
                          "Cache-Control: no-cache\r\nContent-Type: " +
                          content_type + "\r\nContent-Length: " +
                          to_string(text.size()) + "\r\n\r\n";
    return sendAll(fd, header.data(), header.size()) &&
           sendAll(fd, text.data(), text.size());
}

static string requestPath(const string& request) {
    const size_t first = request.find(' ');
    if (first == string::npos) return "/";
    const size_t second = request.find(' ', first + 1);
    if (second == string::npos) return "/";
    return request.substr(first + 1, second - first - 1);
}

static bool queryValue(const string& path, const string& name, string* value) {
    const string key = name + "=";
    const size_t start = path.find(key);
    if (start == string::npos) return false;
    const size_t value_start = start + key.size();
    const size_t end = path.find('&', value_start);
    *value = path.substr(value_start, end == string::npos
                                           ? string::npos : end - value_start);
    return !value->empty();
}

static bool queryInt(const string& path, const string& name, int* value) {
    string text;
    if (!queryValue(path, name, &text)) return false;
    try {
        size_t used = 0;
        const int parsed = stoi(text, &used);
        if (used != text.size()) return false;
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static Rect clampRoi(const RoiSettings& settings) {
    Rect image(0, 0, TrafficConfig::process_width, TrafficConfig::process_height);
    Rect roi(settings.x, settings.y, settings.width, settings.height);
    roi &= image;
    return roi;
}

static RoiSettings currentRoi() {
    lock_guard<mutex> guard(g_roi_lock);
    return g_roi;
}

static ColorDetection detectColor(const Mat& frame, const Rect& roi,
                                  bool red) {
    ColorDetection result;
    if (frame.empty() || roi.width < 2 || roi.height < 2) return result;

    Mat hsv;
    cvtColor(frame(roi), hsv, COLOR_BGR2HSV);
    if (red) {
        Mat low_red;
        Mat high_red;
        inRange(hsv, Scalar(TrafficConfig::red1_h_min,
                            TrafficConfig::red_s_min,
                            TrafficConfig::red_v_min),
                Scalar(TrafficConfig::red1_h_max, 255, 255), low_red);
        inRange(hsv, Scalar(TrafficConfig::red2_h_min,
                            TrafficConfig::red_s_min,
                            TrafficConfig::red_v_min),
                Scalar(TrafficConfig::red2_h_max, 255, 255), high_red);
        bitwise_or(low_red, high_red, result.mask);
    } else {
        inRange(hsv, Scalar(TrafficConfig::green_h_min,
                            TrafficConfig::green_s_min,
                            TrafficConfig::green_v_min),
                Scalar(TrafficConfig::green_h_max, 255, 255), result.mask);
    }

    const Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(result.mask, result.mask, MORPH_OPEN, kernel);
    morphologyEx(result.mask, result.mask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(result.mask.clone(), contours, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    for (const vector<Point>& contour : contours) {
        const double area = contourArea(contour);
        const Rect local = boundingRect(contour);
        if (area < TrafficConfig::color_min_area || local.area() <= 0) continue;
        const double fill_ratio = area / local.area();
        if (fill_ratio < TrafficConfig::color_min_fill_ratio) continue;
        if (!result.found || area > result.area) {
            result.found = true;
            result.area = area;
            result.box = Rect(local.x + roi.x, local.y + roi.y,
                              local.width, local.height);
        }
    }
    return result;
}

class TrafficSequence {
public:
    void reset() {
        state_ = TrafficState::WaitingForRed;
        green_confirm_count_ = 0;
        red_timing_ = false;
    }

    DetectionResult process(const Mat& input, const RoiSettings& settings,
                            chrono::steady_clock::time_point now) {
        DetectionResult result;
        if (input.empty()) return result;
        resize(input, result.raw, Size(TrafficConfig::process_width,
                                       TrafficConfig::process_height),
               0.0, 0.0, INTER_AREA);
        result.roi = clampRoi(settings);
        result.state = state_;
        result.checking_red = state_ == TrafficState::WaitingForRed;

        if (result.roi.width > 1 && result.roi.height > 1 &&
            state_ != TrafficState::Complete) {
            result.color = detectColor(result.raw, result.roi,
                                       result.checking_red);
            if (state_ == TrafficState::WaitingForRed) {
                if (result.color.found) {
                    if (!red_timing_) {
                        red_started_at_ = now;
                        red_timing_ = true;
                    }
                    result.red_elapsed_ms = static_cast<int>(
                        chrono::duration_cast<chrono::milliseconds>(
                            now - red_started_at_).count());
                    if (result.red_elapsed_ms >= TrafficConfig::red_confirm_ms) {
                        state_ = TrafficState::WaitingForGreen;
                        green_confirm_count_ = 0;
                        red_timing_ = false;
                        cout << "TRAFFIC_RED_CONFIRMED after "
                             << TrafficConfig::red_confirm_ms
                             << "ms (integration action: STOP)" << endl;
                    }
                } else {
                    red_timing_ = false;
                    result.red_elapsed_ms = 0;
                }
            } else {
                green_confirm_count_ = result.color.found
                    ? green_confirm_count_ + 1 : 0;
                if (green_confirm_count_ >= TrafficConfig::green_confirm_frames) {
                    state_ = TrafficState::Complete;
                    cout << "TRAFFIC_GREEN_CONFIRMED (integration action: START)" << endl;
                    green_confirm_count_ = 0;
                }
            }
        }

        result.state = state_;
        result.confirm_count = green_confirm_count_;
        result.debug = result.raw.clone();
        rectangle(result.debug, result.roi, Scalar(255, 255, 0), 2, LINE_AA);
        const Scalar color = result.checking_red ? Scalar(0, 0, 255)
                                                  : Scalar(0, 255, 0);
        if (result.color.found) rectangle(result.debug, result.color.box, color,
                                          2, LINE_AA);
        putText(result.debug, string("STATE=") + stateName(result.state),
                Point(10, 24), FONT_HERSHEY_SIMPLEX, 0.60, color, 2, LINE_AA);
        const string progress = result.checking_red
            ? "red_ms=" + to_string(result.red_elapsed_ms) + "/" +
                  to_string(TrafficConfig::red_confirm_ms)
            : "green_confirm=" + to_string(result.confirm_count) + "/" +
                  to_string(TrafficConfig::green_confirm_frames);
        putText(result.debug, string("checking=") +
                (result.checking_red ? "RED" : "GREEN") + " " + progress,
                Point(10, 49), FONT_HERSHEY_SIMPLEX, 0.48,
                Scalar(255, 255, 255), 1, LINE_AA);

        result.mask_frame = Mat::zeros(result.raw.size(), CV_8UC3);
        if (!result.color.mask.empty() && result.roi.width > 1 &&
            result.roi.height > 1) {
            Mat color_mask;
            cvtColor(result.color.mask, color_mask, COLOR_GRAY2BGR);
            result.mask_frame(result.roi) = color_mask;
            rectangle(result.mask_frame, result.roi, color, 2, LINE_AA);
        }
        return result;
    }

private:
    TrafficState state_ = TrafficState::WaitingForRed;
    int green_confirm_count_ = 0;
    bool red_timing_ = false;
    chrono::steady_clock::time_point red_started_at_{};
};

static void publishFrames(RawStreamFrames& frames, const DetectionResult& result) {
    lock_guard<mutex> guard(frames.lock);
    frames.images[kRawView] = result.raw;
    frames.images[kDebugView] = result.debug;
    frames.images[kMaskView] = result.mask_frame;
    ++frames.sequence;
    frames.ready.notify_one();
}

static void runEncoder(RawStreamFrames& raw_frames, StreamFrames& stream_frames) {
    const vector<int> params = {IMWRITE_JPEG_QUALITY, TrafficConfig::jpeg_quality};
    uint64_t consumed = 0;
    while (g_running) {
        array<Mat, kStreamViewCount> images;
        uint64_t source_sequence = 0;
        {
            unique_lock<mutex> lock(raw_frames.lock);
            raw_frames.ready.wait_for(lock, chrono::milliseconds(500), [&] {
                return !g_running || raw_frames.sequence != consumed;
            });
            if (!g_running) break;
            if (raw_frames.sequence == consumed) continue;
            images = raw_frames.images;
            source_sequence = raw_frames.sequence;
            consumed = source_sequence;
        }

        array<vector<uint8_t>, kStreamViewCount> encoded;
        array<bool, kStreamViewCount> updated{};
        bool any_updated = false;
        for (size_t index = 0; index < kStreamViewCount; ++index) {
            if (stream_frames.viewers[index].load(memory_order_relaxed) <= 0 ||
                images[index].empty()) continue;
            updated[index] = imencode(".jpg", images[index], encoded[index], params);
            any_updated = any_updated || updated[index];
        }
        if (!any_updated) continue;
        {
            lock_guard<mutex> guard(stream_frames.lock);
            for (size_t index = 0; index < kStreamViewCount; ++index) {
                if (!updated[index]) continue;
                stream_frames.jpeg[index] = move(encoded[index]);
                stream_frames.sequence[index] = source_sequence;
            }
        }
        stream_frames.ready.notify_all();
    }
}

static size_t streamIndex(const string& view) {
    if (view == "raw") return kRawView;
    if (view == "mask") return kMaskView;
    return kDebugView;
}

static string makeStatusJson() {
    const RoiSettings settings = currentRoi();
    ostringstream json;
    json << "{\"state\":\"" << stateName(static_cast<TrafficState>(
                g_traffic_state.load())) << "\""
         << ",\"confirm\":" << g_confirm_count.load()
         << ",\"green_confirm_required\":"
         << TrafficConfig::green_confirm_frames
         << ",\"red_elapsed_ms\":" << g_red_elapsed_ms.load()
         << ",\"red_confirm_required_ms\":"
         << TrafficConfig::red_confirm_ms
         << ",\"red_area\":" << g_red_area.load()
         << ",\"green_area\":" << g_green_area.load()
         << ",\"x\":" << settings.x
         << ",\"y\":" << settings.y
         << ",\"w\":" << settings.width
         << ",\"h\":" << settings.height << '}';
    return json.str();
}

static string makePage() {
    return R"HTML(<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>第二摄像头红绿灯测试</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111615;color:#e6ece9;font-family:Arial,"Microsoft YaHei",sans-serif;font-size:14px}button,input{font:inherit}button{border:1px solid #3a4543;border-radius:5px;padding:8px 10px;color:#e7eceb;background:#26322f;cursor:pointer}button:hover{background:#33423d}.primary{background:#167d57;border-color:#1b9a6a}.danger{background:#a73838;border-color:#ca5050}.top{min-height:56px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:9px 14px;border-bottom:1px solid #303937;background:#1a201e}.brand{font-weight:700;font-size:16px}.tabs{display:flex;gap:4px;flex-wrap:wrap}.tabs button{padding:6px 9px}.tabs button.active{background:#167d57;border-color:#1b9a6a}.state{margin-left:auto;color:#aebbb7}.state b{color:#e7eceb}.main{min-height:calc(100vh - 56px);display:grid;grid-template-columns:minmax(0,1fr) 310px}.stage{background:#050606;display:flex;align-items:center;justify-content:center;padding:10px}.stage img{display:block;width:100%;height:calc(100vh - 76px);object-fit:contain}.side{background:#171c1b;border-left:1px solid #303937}.section{padding:14px;border-bottom:1px solid #303937}.section h2{font-size:14px;margin:0 0 10px}.grid{display:grid;grid-template-columns:1fr 78px;gap:8px;align-items:center;margin-top:8px}.grid input{min-width:0;width:100%;border:1px solid #3a4543;border-radius:4px;background:#0d1110;color:#f4f7f6;padding:7px}.row{display:flex;gap:8px;margin-top:10px}.row button{flex:1}.hint{margin-top:10px;color:#aebbb7;font-size:12px;line-height:1.45}.notice{color:#96d5b5;min-height:18px}@media(max-width:860px){.main{grid-template-columns:1fr}.side{border-left:0;border-top:1px solid #303937}.stage img{height:auto;max-height:58vh}.state{margin-left:0;width:100%}}
</style></head><body>
<header class="top"><div class="brand">第二摄像头红绿灯测试</div><div class="tabs"><button data-view="debug" class="active">调试图</button><button data-view="raw">原图</button><button data-view="mask">颜色掩膜</button></div><div class="state">状态：<b id="state">WAIT_RED</b></div></header>
<main class="main"><section class="stage"><img id="stream" src="/stream?view=debug" alt="红绿灯图传"></section><aside class="side"><section class="section"><h2>固定红绿灯 ROI</h2><div class="grid"><label for="x">X</label><input id="x" type="number" min="0" max="639"></div><div class="grid"><label for="y">Y</label><input id="y" type="number" min="0" max="359"></div><div class="grid"><label for="w">宽</label><input id="w" type="number" min="1" max="640"></div><div class="grid"><label for="h">高</label><input id="h" type="number" min="1" max="360"></div><div class="row"><button id="apply" class="primary">应用 ROI</button><button id="reset" class="danger">重置红绿灯状态</button></div><div class="hint">坐标以 640x360 调试图为准。黄色框是唯一参与颜色检测的区域。</div></section><section class="section"><h2>检测状态</h2><div id="detail" class="hint"></div><div id="notice" class="notice"></div></section></aside></main>
<script>
const img=document.getElementById('stream'),notice=document.getElementById('notice');const ids=['x','y','w','h'];
function note(text,bad=false){notice.textContent=text;notice.style.color=bad?'#ffaaa2':'#96d5b5';setTimeout(()=>{if(notice.textContent===text)notice.textContent=''},2600)}
async function post(url){try{const r=await fetch(url,{method:'POST'});const t=await r.text();if(!r.ok)throw Error(t);note(t);refresh()}catch(e){note(e.message||'请求失败',true)}}
function show(data){document.getElementById('state').textContent=data.state||'WAIT_RED';for(const id of ids){const input=document.getElementById(id);if(document.activeElement!==input&&data[id]!==undefined)input.value=data[id]}const progress=data.state==='WAIT_RED'?('红灯持续 '+(data.red_elapsed_ms||0)+'/'+(data.red_confirm_required_ms||3000)+' ms'):('绿灯确认 '+(data.confirm||0)+'/'+(data.green_confirm_required||10));document.getElementById('detail').textContent=progress+'；红色面积 '+(data.red_area||0)+'；绿色面积 '+(data.green_area||0)}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(r.ok)show(await r.json())}catch(e){}}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));b.classList.add('active');img.src='/stream?view='+b.dataset.view+'&t='+Date.now()});
document.getElementById('apply').onclick=()=>{const q=new URLSearchParams();ids.forEach(id=>q.set(id,document.getElementById(id).value));post('/api/roi?'+q.toString())};
document.getElementById('reset').onclick=()=>post('/api/reset');refresh();setInterval(refresh,500);
</script></body></html>)HTML";
}

static void streamCamera(int client_fd, StreamFrames& frames, size_t index) {
    timeval timeout{2, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    const int no_delay = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    const int buffer_size = 64 * 1024;
    setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    const string header = "HTTP/1.1 200 OK\r\nConnection: close\r\n"
                          "Cache-Control: no-cache, private\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!sendAll(client_fd, header.data(), header.size())) {
        close(client_fd);
        return;
    }

    frames.viewers[index].fetch_add(1, memory_order_relaxed);
    uint64_t sent = 0;
    {
        lock_guard<mutex> guard(frames.lock);
        sent = frames.sequence[index];
    }
    while (g_running) {
        vector<uint8_t> jpeg;
        {
            unique_lock<mutex> lock(frames.lock);
            frames.ready.wait_for(lock, chrono::milliseconds(500), [&] {
                return !g_running || frames.sequence[index] != sent;
            });
            if (!g_running) break;
            if (frames.sequence[index] == sent) continue;
            jpeg = frames.jpeg[index];
            sent = frames.sequence[index];
        }
        if (jpeg.empty()) continue;
        const string part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                            to_string(jpeg.size()) + "\r\n\r\n";
        if (!sendAll(client_fd, part.data(), part.size()) ||
            !sendAll(client_fd, jpeg.data(), jpeg.size()) ||
            !sendAll(client_fd, "\r\n", 2)) break;
    }
    frames.viewers[index].fetch_sub(1, memory_order_relaxed);
    close(client_fd);
}

static void handleClient(int client_fd, StreamFrames& frames) {
    timeval timeout{2, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char request[2048] = {0};
    const ssize_t received = recv(client_fd, request, sizeof(request) - 1, 0);
    if (received <= 0) {
        close(client_fd);
        return;
    }
    const string path = requestPath(request);
    if (path == "/" || path.rfind("/?", 0) == 0) {
        sendText(client_fd, makePage(), "text/html; charset=utf-8");
        close(client_fd);
        return;
    }
    if (path == "/api/status") {
        sendText(client_fd, makeStatusJson(), "application/json; charset=utf-8");
        close(client_fd);
        return;
    }
    if (path == "/api/reset") {
        g_reset_requested = true;
        sendText(client_fd, "红绿灯状态已重置，重新等待红灯", "text/plain; charset=utf-8");
        close(client_fd);
        return;
    }
    if (path.rfind("/api/roi", 0) == 0) {
        RoiSettings settings;
        if (!queryInt(path, "x", &settings.x) ||
            !queryInt(path, "y", &settings.y) ||
            !queryInt(path, "w", &settings.width) ||
            !queryInt(path, "h", &settings.height) ||
            clampRoi(settings).area() < 4) {
            sendText(client_fd, "ROI 参数无效", "text/plain; charset=utf-8");
        } else {
            const Rect roi = clampRoi(settings);
            settings.x = roi.x;
            settings.y = roi.y;
            settings.width = roi.width;
            settings.height = roi.height;
            {
                lock_guard<mutex> guard(g_roi_lock);
                g_roi = settings;
            }
            g_reset_requested = true;
            sendText(client_fd, "ROI 已应用，红绿灯状态已重置",
                     "text/plain; charset=utf-8");
        }
        close(client_fd);
        return;
    }
    if (path.rfind("/stream", 0) != 0) {
        sendText(client_fd, "not found", "text/plain; charset=utf-8");
        close(client_fd);
        return;
    }
    string view = "debug";
    queryValue(path, "view", &view);
    streamCamera(client_fd, frames, streamIndex(view));
}

static void runServer(int port, const string& host, StreamFrames& frames,
                      vector<thread>& clients, mutex& clients_lock) {
    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        cerr << "HTTP socket creation failed" << endl;
        g_running = false;
        return;
    }
    int option = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(server_fd, 8) < 0) {
        cerr << "HTTP bind/listen failed, port=" << port << endl;
        close(server_fd);
        g_running = false;
        return;
    }
    const string url = "http://" + host + ":" + to_string(port) + "/";
    cout << "Traffic-light stream: " << url << endl;
    while (g_running) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(server_fd, &set);
        timeval timeout{1, 0};
        const int ready = select(server_fd + 1, &set, nullptr, nullptr, &timeout);
        if (ready <= 0 || !FD_ISSET(server_fd, &set)) continue;
        const int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        lock_guard<mutex> guard(clients_lock);
        clients.emplace_back(handleClient, client_fd, ref(frames));
    }
    close(server_fd);
}

int main(int argc, char** argv) {
    const string device = argument(argc, argv, "--device", "/dev/video2");
    const int port = stoi(argument(argc, argv, "--port", "8092"));
    const string host = argument(argc, argv, "--ip", "192.168.137.161");

    VideoCapture camera;
#ifdef __linux__
    camera.open(device, CAP_V4L2);
#else
    camera.open(device);
#endif
    if (!camera.isOpened()) {
        cerr << "Cannot open second camera: " << device << endl;
        return 1;
    }
    camera.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera.set(CAP_PROP_FRAME_WIDTH, TrafficConfig::capture_width);
    camera.set(CAP_PROP_FRAME_HEIGHT, TrafficConfig::capture_height);
    camera.set(CAP_PROP_FPS, 25);
    camera.set(CAP_PROP_BUFFERSIZE, 1);
    cout << "Second camera actual mode: "
         << camera.get(CAP_PROP_FRAME_WIDTH) << "x"
         << camera.get(CAP_PROP_FRAME_HEIGHT) << " @"
         << camera.get(CAP_PROP_FPS) << " "
         << fourccName(static_cast<int>(camera.get(CAP_PROP_FOURCC))) << endl;

    signal(SIGINT, stopOnSignal);
    signal(SIGTERM, stopOnSignal);
    TrafficSequence sequence;
    RawStreamFrames raw_frames;
    StreamFrames stream_frames;
    vector<thread> clients;
    mutex clients_lock;
    thread encoder(runEncoder, ref(raw_frames), ref(stream_frames));
    thread server(runServer, port, host, ref(stream_frames), ref(clients),
                  ref(clients_lock));

    TrafficState last_state = TrafficState::WaitingForRed;
    while (g_running) {
        Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            cerr << "Second camera frame read failed" << endl;
            break;
        }
        if (g_reset_requested.exchange(false)) sequence.reset();
        const DetectionResult result = sequence.process(
            frame, currentRoi(), chrono::steady_clock::now());
        g_traffic_state = static_cast<int>(result.state);
        g_confirm_count = result.confirm_count;
        g_red_elapsed_ms = result.red_elapsed_ms;
        if (result.checking_red) {
            g_red_area = static_cast<int>(lround(result.color.area));
            g_green_area = 0;
        } else {
            g_red_area = 0;
            g_green_area = static_cast<int>(lround(result.color.area));
        }
        if (result.state != last_state) {
            cout << "TRAFFIC_STATE=" << stateName(result.state) << endl;
            last_state = result.state;
        }
        publishFrames(raw_frames, result);
    }

    g_running = false;
    raw_frames.ready.notify_all();
    stream_frames.ready.notify_all();
    if (encoder.joinable()) encoder.join();
    if (server.joinable()) server.join();
    {
        lock_guard<mutex> guard(clients_lock);
        for (thread& client : clients) {
            if (client.joinable()) client.join();
        }
    }
    camera.release();
    return 0;
}
