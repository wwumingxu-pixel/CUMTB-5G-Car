// 更新说明：新增 1920x1080 摄像头采集与 320x180 缩放输出，图传线程只保留最新 JPEG，减少网络延迟。
// 更新日期：2026-09
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_running(true);

constexpr int kDefaultPort = 8090;
constexpr int kDefaultCameraId = 0;
constexpr int kCaptureWidth = 1920;
constexpr int kCaptureHeight = 1080;
constexpr int kDefaultWidth = 320;
constexpr int kDefaultHeight = 180;
constexpr int kDefaultJpegQuality = 45;
constexpr int kDefaultFps = 25;
constexpr const char* kPiIp = "192.168.137.161";

struct LatestJpegFrame {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<uint8_t> bytes;
    uint64_t sequence = 0;
};

void onSignal(int) {
    g_running = false;
}

bool parseInt(const char* s, int& out) {
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

bool sendAll(int fd, const void* data, size_t len) {
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

bool sendString(int fd, const std::string& s) {
    return sendAll(fd, s.data(), s.size());
}

std::string makeHtmlPage() {
    return R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>智能车图传</title>
<style>
    * { box-sizing: border-box; }
    body {
        margin: 0;
        background: #0f172a;
        color: #e5e7eb;
        font-family: Arial, "Microsoft YaHei", sans-serif;
        overflow: hidden;
    }
    .bar {
        height: 58px;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: rgba(15, 23, 42, 0.96);
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    }
    .title {
        font-size: 18px;
        font-weight: 700;
        margin-right: auto;
        white-space: nowrap;
    }
    button {
        border: 0;
        border-radius: 10px;
        padding: 10px 14px;
        color: white;
        background: #2563eb;
        font-size: 15px;
        cursor: pointer;
    }
    button:active { transform: scale(0.97); }
    #recordBtn.recording { background: #dc2626; }
    #stage {
        width: 100vw;
        height: calc(100vh - 58px);
        display: flex;
        align-items: center;
        justify-content: center;
        background: #020617;
    }
    #stream {
        width: 100%;
        height: 100%;
        object-fit: contain;
        image-rendering: auto;
        background: #000;
    }
    #tip {
        position: fixed;
        left: 12px;
        bottom: 12px;
        padding: 8px 10px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.75);
        color: #cbd5e1;
        font-size: 13px;
    }
    @media (max-width: 520px) {
        .title { font-size: 15px; }
        button { padding: 9px 10px; font-size: 14px; }
    }
</style>
</head>
<body>
<div class="bar">
    <div class="title">智能车 HTTP 图传</div>
    <button id="shotBtn">截图</button>
    <button id="recordBtn">开始录屏</button>
    <button id="fullBtn">全屏</button>
</div>
<div id="stage">
    <img id="stream" src="/stream" alt="stream">
</div>
<div id="tip">画面自适应窗口；截图保存 PNG；录屏保存 WEBM。</div>
<canvas id="canvas" style="display:none"></canvas>
<script>
const img = document.getElementById('stream');
const canvas = document.getElementById('canvas');
const shotBtn = document.getElementById('shotBtn');
const recordBtn = document.getElementById('recordBtn');
const fullBtn = document.getElementById('fullBtn');

let recorder = null;
let chunks = [];
let drawTimer = null;

function timeName(prefix, ext) {
    const d = new Date();
    const pad = n => String(n).padStart(2, '0');
    return `${prefix}_${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.${ext}`;
}

function drawFrame() {
    const w = img.naturalWidth || img.clientWidth || 320;
    const h = img.naturalHeight || img.clientHeight || 240;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
}

function downloadBlob(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        URL.revokeObjectURL(a.href);
        a.remove();
    }, 500);
}

shotBtn.onclick = () => {
    drawFrame();
    canvas.toBlob(blob => {
        if (blob) downloadBlob(blob, timeName('smartcar_shot', 'png'));
    }, 'image/png');
};

recordBtn.onclick = () => {
    if (recorder && recorder.state === 'recording') {
        recorder.stop();
        return;
    }
    drawFrame();
    const stream = canvas.captureStream(20);
    chunks = [];
    recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    recorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
        clearInterval(drawTimer);
        drawTimer = null;
        recordBtn.textContent = '开始录屏';
        recordBtn.classList.remove('recording');
        downloadBlob(new Blob(chunks, { type: 'video/webm' }), timeName('smartcar_record', 'webm'));
    };
    drawTimer = setInterval(drawFrame, 50);
    recorder.start();
    recordBtn.textContent = '停止录屏';
    recordBtn.classList.add('recording');
};

fullBtn.onclick = () => {
    const el = document.documentElement;
    if (!document.fullscreenElement) el.requestFullscreen?.();
    else document.exitFullscreen?.();
};
</script>
</body>
</html>)HTML";
}

bool sendHtmlPage(int fd) {
    std::string body = makeHtmlPage();
    std::string header =
        "HTTP/1.0 200 OK\r\n"
        "Server: smartcar-mjpeg\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    return sendString(fd, header) && sendString(fd, body);
}

void printOpenUrl(int port) {
    std::string url = "http://" + std::string(kPiIp) + ":" + std::to_string(port) + "/";
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "图传网页地址：" << url << std::endl;
    std::cout << "MobaXterm 中可按 Ctrl 后点击，或直接选中复制" << std::endl;
    std::cout << "Clickable: \033]8;;" << url << "\033\\" << url << "\033]8;;\033\\" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
}

std::string parsePath(const char* request_buf) {
    std::string req(request_buf ? request_buf : "");
    size_t first_space = req.find(' ');
    if (first_space == std::string::npos) return "/";
    size_t second_space = req.find(' ', first_space + 1);
    if (second_space == std::string::npos) return "/";
    return req.substr(first_space + 1, second_space - first_space - 1);
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [port] [camera_id] [stream_width] [stream_height]"
              << " [jpeg_quality] [fps]" << std::endl;
    std::cout << "Camera capture is fixed at 1920x1080; width and height"
              << " control the resized MJPEG output." << std::endl;
    std::cout << "Example: " << prog << " 8080 0 320 180 45 25" << std::endl;
    std::cout << "Open browser: http://<raspberry_pi_ip>:8080/" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    int port = kDefaultPort;
    int camera_id = kDefaultCameraId;
    int width = kDefaultWidth;
    int height = kDefaultHeight;
    int jpeg_quality = kDefaultJpegQuality;
    int fps = kDefaultFps;

    if (argc >= 2 && !parseInt(argv[1], port)) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 3 && !parseInt(argv[2], camera_id)) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 4 && !parseInt(argv[3], width)) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 5 && !parseInt(argv[4], height)) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 6 && !parseInt(argv[5], jpeg_quality)) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc >= 7 && !parseInt(argv[6], fps)) {
        printUsage(argv[0]);
        return 1;
    }

    if (width <= 0 || height <= 0) {
        std::cerr << "Stream width and height must be positive" << std::endl;
        return 1;
    }
    if (jpeg_quality < 10) jpeg_quality = 10;
    if (jpeg_quality > 95) jpeg_quality = 95;
    if (fps < 5) fps = 5;
    if (fps > 60) fps = 60;

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    cv::VideoCapture cap(camera_id, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera: " << camera_id << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, kCaptureWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kCaptureHeight);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create TCP socket" << std::endl;
        return 1;
    }

    int opt = 1;
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed, port=" << port << std::endl;
        ::close(server_fd);
        return 1;
    }

    if (::listen(server_fd, 1) < 0) {
        std::cerr << "Listen failed" << std::endl;
        ::close(server_fd);
        return 1;
    }

    std::cout << "HTTP MJPEG stream started on port " << port << std::endl;
    std::cout << "Open: http://<raspberry_pi_ip>:" << port << "/" << std::endl;
    printOpenUrl(port);
    const int actual_fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    std::string actual_format(4, ' ');
    for (int index = 0; index < 4; ++index) {
        actual_format[index] = static_cast<char>(
            (actual_fourcc >> (8 * index)) & 0xff);
    }
    std::cout << "Camera actual mode: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << " FPS "
              << actual_format << std::endl;
    std::cout << "Params: cam=" << camera_id
              << " capture=" << kCaptureWidth << "x" << kCaptureHeight
              << " stream=" << width << "x" << height
              << " q=" << jpeg_quality << " fps=" << fps << std::endl;

    const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    LatestJpegFrame latest_frame;

    // Capture the full-view sensor mode, resize the whole frame for MJPEG, and
    // keep only the newest encoded frame when a network client is slow.
    std::thread capture_thread([&] {
        cv::Mat frame;
        cv::Mat stream_frame;
        std::vector<uint8_t> jpg;

        while (g_running) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "Camera read failed" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            cv::resize(frame, stream_frame, cv::Size(width, height),
                       0.0, 0.0, cv::INTER_AREA);
            if (!cv::imencode(".jpg", stream_frame, jpg, jpeg_params)) {
                std::cerr << "JPEG encode failed" << std::endl;
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(latest_frame.mutex);
                latest_frame.bytes = jpg;
                ++latest_frame.sequence;
            }
            latest_frame.ready.notify_all();
        }
    });

    while (g_running) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (g_running) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }

        char client_ip[64] = {0};
        ::inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cout << "Client connected: " << client_ip << std::endl;

        char request_buf[1024] = {0};
        ::recv(client_fd, request_buf, sizeof(request_buf) - 1, 0);
        std::string path = parsePath(request_buf);

        if (path != "/stream") {
            sendHtmlPage(client_fd);
            ::close(client_fd);
            continue;
        }

        const int no_delay = 1;
        ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
        // A small socket buffer avoids accumulating a long queue of old MJPEG frames.
        const int send_buffer_bytes = 64 * 1024;
        ::setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF,
                     &send_buffer_bytes, sizeof(send_buffer_bytes));

        std::string header =
            "HTTP/1.0 200 OK\r\n"
            "Server: smartcar-mjpeg\r\n"
            "Connection: close\r\n"
            "Max-Age: 0\r\n"
            "Expires: 0\r\n"
            "Cache-Control: no-cache, private\r\n"
            "Pragma: no-cache\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

        if (!sendString(client_fd, header)) {
            ::close(client_fd);
            continue;
        }

        uint64_t sent_sequence = 0;
        while (g_running) {
            std::vector<uint8_t> jpg;
            {
                std::unique_lock<std::mutex> lock(latest_frame.mutex);
                latest_frame.ready.wait_for(lock, std::chrono::milliseconds(500), [&] {
                    return !g_running || latest_frame.sequence != sent_sequence;
                });
                if (!g_running) break;
                if (latest_frame.sequence == sent_sequence || latest_frame.bytes.empty()) {
                    continue;
                }
                jpg = latest_frame.bytes;
                sent_sequence = latest_frame.sequence;
            }

            std::string part_header =
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " + std::to_string(jpg.size()) + "\r\n\r\n";

            if (!sendString(client_fd, part_header)) break;
            if (!sendAll(client_fd, jpg.data(), jpg.size())) break;
            if (!sendString(client_fd, "\r\n")) break;
        }

        std::cout << "Client disconnected" << std::endl;
        ::close(client_fd);
    }

    g_running = false;
    latest_frame.ready.notify_all();
    if (capture_thread.joinable()) capture_thread.join();
    ::close(server_fd);
    cap.release();
    return 0;
}
