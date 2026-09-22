// 更新说明：网页键控支持 W/A/S/D、方向键和网页按钮控制前进/后退/转向，可分别输入前进/后退 PWM；松键或失联自动停车。
// 摄像头先采集 1920x1080，再缩放到 320x180 输出 MJPEG。
// 更新日期：2026-09

#include <pigpio.h>
#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kMotorPwmPin = 13;       // BCM GPIO13, physical pin 33
constexpr int kServoPin = 12;          // BCM GPIO12, physical pin 32
// Keep these identical to mada.cpp, which is verified on this car.
constexpr int kMotorPwmRange = 40000;
constexpr int kMotorPwmFrequency = 200;
constexpr int kMotorDrivePwm = 11000;  // 前进：>10000，实测 11000 前进正常。
constexpr int kMotorReversePwm = 9000; // 后退：<10000，9500 开始倒退，取 9000 稳定后退。
constexpr int kMotorStopPwm = 10000;   // 停止：中位。
constexpr auto kReverseBrakePause = std::chrono::milliseconds(300);  // 前进→后退前回中位刹车的停留时间。
constexpr int kServoCenter = 85;    // 中位：官方文档初始化 85 度（若实测回正应 90 度则改回 90）。
constexpr int kServoMin = 50;       // 官方扫描行程下限 50 度。
constexpr int kServoMax = 150;      // 官方扫描行程上限 150 度。
constexpr int kTurnDegrees = 25;    // 转向幅度（度），可按需调整。
constexpr int kServoDirection = -1; // 左右反了改成 1。
constexpr int kDefaultPort = 8090;
constexpr int kDefaultCamera = 0;
constexpr int kCaptureWidth = 1920;
constexpr int kCaptureHeight = 1080;
constexpr int kStreamWidth = 320;
constexpr int kStreamHeight = 180;
constexpr int kStreamFps = 24;
constexpr int kJpegQuality = 55;
constexpr auto kWebControlTimeout = std::chrono::milliseconds(700);

std::atomic<bool> g_running(true);
std::atomic<int> g_drive_state(0);     // -1 reverse, 0 stop, 1 forward
std::atomic<int> g_steering_state(0); // -1 left, 0 center, 1 right
std::atomic<int> g_motor_pwm(0);
volatile sig_atomic_t g_stop_requested = 0;

struct LatestFrame {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<uchar> jpeg;
    uint64_t sequence = 0;
};

LatestFrame g_latest_frame;

struct WebControlState {
    std::mutex mutex;
    int drive = 0;
    int steering = 0;
    int forward_pwm = kMotorDrivePwm;
    int reverse_pwm = kMotorReversePwm;
    bool has_heartbeat = false;
    std::chrono::steady_clock::time_point last_update{};
};

WebControlState g_web_control;

void onSignal(int) {
    g_stop_requested = 1;
}

int clampInt(int value, int low, int high) {
    return std::max(low, std::min(high, value));
}

bool parseInt(const char* text, int& value) {
    try {
        size_t used = 0;
        const int parsed = std::stoi(text, &used);
        if (text[used] != '\0') return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

class MotorDriver {
public:
    // Set both direction pins to non-negative BCM GPIO numbers for an H-bridge.
    // With the existing one-wire controller, leave them as -1.
    MotorDriver(int pwm_pin, int direction_a_pin, int direction_b_pin)
        : pwm_pin_(pwm_pin), direction_a_pin_(direction_a_pin), direction_b_pin_(direction_b_pin) {}

    bool initialize() {
        if (gpioSetMode(pwm_pin_, PI_OUTPUT) < 0) return false;
        // 顺序：先设频率，再设范围（此顺序下 duty 才有 0~40000 区分度，10000 为中位）。
        const int actual_frequency = gpioSetPWMfrequency(pwm_pin_, kMotorPwmFrequency);
        const int actual_range = gpioSetPWMrange(pwm_pin_, kMotorPwmRange);
        std::cout << "Motor BCM GPIO=" << pwm_pin_
                  << " (physical pin 33), range=" << actual_range
                  << ", frequency=" << actual_frequency << " Hz" << std::endl;

        if ((direction_a_pin_ >= 0) != (direction_b_pin_ >= 0)) {
            std::cerr << "Direction GPIOs must be configured as a pair." << std::endl;
            return false;
        }
        has_direction_pins_ = direction_a_pin_ >= 0;
        if (has_direction_pins_) {
            if (gpioSetMode(direction_a_pin_, PI_OUTPUT) < 0 ||
                gpioSetMode(direction_b_pin_, PI_OUTPUT) < 0) {
                return false;
            }
            gpioWrite(direction_a_pin_, 0);
            gpioWrite(direction_b_pin_, 0);
        }
        stop();
        return true;
    }

    bool drive(int direction, int pwm) {
        direction = clampInt(direction, -1, 1);
        if (direction == 0) {
            stop();
            return true;
        }

        if (has_direction_pins_) {
            // Remove PWM before changing H-bridge direction to avoid a transient drive pulse.
            gpioPWM(pwm_pin_, 0);
            gpioWrite(direction_a_pin_, direction > 0 ? 1 : 0);
            gpioWrite(direction_b_pin_, direction > 0 ? 0 : 1);
            setPwm(pwm);
            return true;
        }

        // 单路双向电调：方向由 PWM 值本身区分，直接输出调用者给定的前进/后退 PWM。
        setPwm(pwm);
        return true;
    }

    void setPwm(int pwm) {
        pwm = clampInt(pwm, 0, kMotorPwmRange);
        const int result = gpioPWM(pwm_pin_, pwm);
        if (result < 0) {
            std::cerr << "gpioPWM failed, code=" << result << std::endl;
        }
    }

    void stop() {
        // 停车用中位 10000，而不是 0：duty 40~50 是后退，0 附近可能不安全。
        setPwm(kMotorStopPwm);
        if (has_direction_pins_) {
            gpioWrite(direction_a_pin_, 0);
            gpioWrite(direction_b_pin_, 0);
        }
    }

    bool supportsReverse() const { return true; }  // 单路双向电调或 H 桥均可后退。

private:
    int pwm_pin_;
    int direction_a_pin_;
    int direction_b_pin_;
    bool has_direction_pins_ = false;
};

class SteeringServo {
public:
    bool initialize() {
        if (gpioSetMode(kServoPin, PI_OUTPUT) < 0) return false;
        current_angle_ = kServoCenter;
        target_angle_ = kServoCenter;
        writeAngle(current_angle_);
        return true;
    }

    void setDirection(int direction) {
        direction = clampInt(direction, -1, 1);
        target_angle_ = clampInt(kServoCenter + kServoDirection * direction * kTurnDegrees,
                                 kServoMin, kServoMax);
    }

    void center() { target_angle_ = kServoCenter; }

    // Move a small amount per control cycle so steering changes are not abrupt.
    void update() {
        constexpr int kStepDegrees = 2;
        if (current_angle_ < target_angle_) {
            current_angle_ = std::min(current_angle_ + kStepDegrees, target_angle_);
        } else if (current_angle_ > target_angle_) {
            current_angle_ = std::max(current_angle_ - kStepDegrees, target_angle_);
        } else {
            return;
        }
        writeAngle(current_angle_);
    }

    void centerImmediately() {
        target_angle_ = kServoCenter;
        current_angle_ = kServoCenter;
        writeAngle(current_angle_);
    }

private:
    static int angleToPulseWidth(int angle) {
        return 500 + (clampInt(angle, 0, 180) * 2000) / 180;
    }

    static void writeAngle(int angle) {
        gpioServo(kServoPin, angleToPulseWidth(angle));
    }

    int current_angle_ = kServoCenter;
    int target_angle_ = kServoCenter;
};

class TerminalInput {
public:
    bool open() {
        if (!isatty(STDIN_FILENO)) {
            std::cerr << "Standard input is not a terminal; keyboard control is unavailable." << std::endl;
            return false;
        }
        if (tcgetattr(STDIN_FILENO, &old_termios_) != 0) return false;
        old_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (old_flags_ < 0) return false;

        termios raw = old_termios_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0 ||
            fcntl(STDIN_FILENO, F_SETFL, old_flags_ | O_NONBLOCK) != 0) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
            return false;
        }
        active_ = true;
        return true;
    }

    ~TerminalInput() { close(); }

    void close() {
        if (!active_) return;
        tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
        fcntl(STDIN_FILENO, F_SETFL, old_flags_);
        active_ = false;
    }

private:
    termios old_termios_{};
    int old_flags_ = 0;
    bool active_ = false;
};

bool sendAll(int fd, const void* data, size_t length) {
    const char* bytes = static_cast<const char*>(data);
    while (length > 0) {
        const ssize_t sent = ::send(fd, bytes, length, MSG_NOSIGNAL);
        if (sent <= 0) return false;
        bytes += sent;
        length -= static_cast<size_t>(sent);
    }
    return true;
}

bool sendText(int fd, const std::string& text) {
    return sendAll(fd, text.data(), text.size());
}

std::string requestPath(const char* request) {
    const std::string line(request == nullptr ? "" : request);
    const size_t first_space = line.find(' ');
    if (first_space == std::string::npos) return "/";
    const size_t second_space = line.find(' ', first_space + 1);
    if (second_space == std::string::npos) return "/";
    return line.substr(first_space + 1, second_space - first_space - 1);
}

bool parseQueryInt(const std::string& path, const std::string& name, int& value) {
    const std::string key = name + "=";
    const size_t begin = path.find(key);
    if (begin == std::string::npos) return false;
    const size_t value_begin = begin + key.size();
    const size_t value_end = path.find('&', value_begin);
    try {
        size_t used = 0;
        const std::string text = path.substr(value_begin, value_end - value_begin);
        const int parsed = std::stoi(text, &used);
        if (used != text.size()) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool updateWebControl(const std::string& path) {
    int drive = 0;
    int steering = 0;
    int forward_pwm = 0;
    int reverse_pwm = 0;
    if (!parseQueryInt(path, "drive", drive) || !parseQueryInt(path, "steer", steering) ||
        !parseQueryInt(path, "fpwm", forward_pwm) || !parseQueryInt(path, "rpwm", reverse_pwm)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_web_control.mutex);
    g_web_control.drive = clampInt(drive, -1, 1);
    g_web_control.steering = clampInt(steering, -1, 1);
    g_web_control.forward_pwm = clampInt(forward_pwm, 0, kMotorPwmRange);
    g_web_control.reverse_pwm = clampInt(reverse_pwm, 0, kMotorPwmRange);
    g_web_control.has_heartbeat = true;
    g_web_control.last_update = std::chrono::steady_clock::now();
    return true;
}

struct WebControlSnapshot {
    int drive = 0;
    int steering = 0;
    int forward_pwm = kMotorDrivePwm;
    int reverse_pwm = kMotorReversePwm;
    bool active = false;
};

WebControlSnapshot currentWebControl() {
    std::lock_guard<std::mutex> lock(g_web_control.mutex);
    WebControlSnapshot state;
    state.drive = g_web_control.drive;
    state.steering = g_web_control.steering;
    state.forward_pwm = g_web_control.forward_pwm;
    state.reverse_pwm = g_web_control.reverse_pwm;
    state.active = g_web_control.has_heartbeat &&
        std::chrono::steady_clock::now() - g_web_control.last_update <= kWebControlTimeout;
    return state;
}

std::string htmlPage() {
    return R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智能车控制与图传</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#e5e7eb;font-family:Arial,"Microsoft YaHei",sans-serif}header{height:52px;display:flex;align-items:center;padding:0 16px;background:#1f2937;font-weight:700}main{height:calc(100vh - 52px);display:grid;grid-template-columns:minmax(0,1fr) 180px;gap:12px;padding:12px;background:#0b1120}.stream{min-width:0;display:grid;place-items:center;background:#000}.stream img{max-width:100%;max-height:calc(100vh - 76px);object-fit:contain}.controls{display:grid;align-content:center;justify-content:center;gap:8px}.pwm{display:grid;gap:4px;font-size:12px;color:#cbd5e1}.pwm input{width:168px;height:36px;border:1px solid #64748b;border-radius:6px;background:#0f172a;color:#f8fafc;padding:0 8px;font-size:16px}.pad{display:grid;grid-template-columns:repeat(3,52px);gap:6px}.key{width:52px;height:52px;border:1px solid #64748b;border-radius:6px;background:#1e293b;color:#f8fafc;font-size:24px;font-weight:700}.key:active,.key.active{background:#2563eb;border-color:#60a5fa}.stop{grid-column:1/4;height:42px;border:0;border-radius:6px;background:#dc2626;color:#fff;font-size:14px;font-weight:700}.state{font-size:12px;color:#cbd5e1;text-align:center;min-height:18px}@media(max-width:680px){main{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}.stream img{max-height:calc(100vh - 300px)}.controls{padding-bottom:4px}}
</style>
</head>
<body>
<header>智能车 HTTP 控制与图传</header>
<main>
  <div class="stream"><img src="/stream" alt="实时图传"></div>
  <section class="controls" aria-label="车辆控制">
    <label class="pwm" for="fpwm">前进 PWM
      <input id="fpwm" type="number" min="0" max="40000" step="100" value="11000" inputmode="numeric">
    </label>
    <label class="pwm" for="rpwm">后退 PWM
      <input id="rpwm" type="number" min="0" max="40000" step="100" value="9000" inputmode="numeric">
    </label>
    <div id="state" class="state">已连接</div>
    <div class="pad">
      <span></span><button class="key" data-key="ArrowUp" aria-label="前进">^</button><span></span>
      <button class="key" data-key="ArrowLeft" aria-label="左转">&lt;</button><button class="key" data-key="ArrowDown" aria-label="后退">v</button><button class="key" data-key="ArrowRight" aria-label="右转">&gt;</button>
      <button id="stop" class="stop">STOP</button>
    </div>
  </section>
</main>
<script>
const pressed=new Set();
const buttons=new Map([...document.querySelectorAll('[data-key]')].map(b=>[b.dataset.key,b]));
const state=document.getElementById('state');
const fpwmInput=document.getElementById('fpwm');
const rpwmInput=document.getElementById('rpwm');
const isInput=e=>e.target===fpwmInput||e.target===rpwmInput;
const accepted=new Set(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','KeyW','KeyS','KeyA','KeyD']);
function has(...keys){return keys.some(k=>pressed.has(k));}
const clampPwm=n=>Math.max(0,Math.min(40000,Math.round(Number(n)||0)));
function values(){const up=has('ArrowUp','KeyW'),down=has('ArrowDown','KeyS'),left=has('ArrowLeft','KeyA'),right=has('ArrowRight','KeyD'),fpwm=clampPwm(fpwmInput.value),rpwm=clampPwm(rpwmInput.value);return{drive:up===down?0:up?1:-1,steer:left===right?0:right?1:-1,fpwm,rpwm};}
function render(){const v=values();buttons.forEach((b,k)=>b.classList.toggle('active',pressed.has(k)));const motion=v.drive===1?'前进':v.drive===-1?'后退':v.steer===-1?'左转':v.steer===1?'右转':'已停止';const cur=v.drive===1?v.fpwm:v.drive===-1?v.rpwm:0;state.textContent=`${motion}  PWM ${cur}`;}
function send(){const v=values();render();fetch(`/control?drive=${v.drive}&steer=${v.steer}&fpwm=${v.fpwm}&rpwm=${v.rpwm}&t=${Date.now()}`,{cache:'no-store'}).catch(()=>{state.textContent='连接中断';});}
function releaseAll(){pressed.clear();send();}
addEventListener('keydown',e=>{if(isInput(e))return;if(!accepted.has(e.code)&&e.code!=='Space')return;e.preventDefault();if(e.code==='Space'){releaseAll();return;}pressed.add(e.code);send();});
addEventListener('keyup',e=>{if(isInput(e)||!accepted.has(e.code))return;e.preventDefault();pressed.delete(e.code);send();});
addEventListener('blur',releaseAll);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseAll();});
[fpwmInput,rpwmInput].forEach(inp=>{inp.addEventListener('change',send);inp.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();inp.blur();send();}});});
setInterval(send,250);
buttons.forEach((button,key)=>{button.addEventListener('pointerdown',e=>{e.preventDefault();pressed.add(key);send();});['pointerup','pointercancel','pointerleave'].forEach(type=>button.addEventListener(type,e=>{e.preventDefault();pressed.delete(key);send();}));});
document.getElementById('stop').addEventListener('click',releaseAll);send();
</script>
</body>
</html>)HTML";
}

void sendPage(int client_fd) {
    const std::string body = htmlPage();
    const std::string header =
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
        "Cache-Control: no-store\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nConnection: close\r\n\r\n";
    sendText(client_fd, header);
    sendText(client_fd, body);
}

void handleStreamClient(int client_fd) {
    timeval timeout{2, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    const std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (!sendText(client_fd, header)) {
        ::close(client_fd);
        return;
    }

    uint64_t sent_sequence = 0;
    while (g_running) {
        std::vector<uchar> jpeg;
        {
            std::unique_lock<std::mutex> lock(g_latest_frame.mutex);
            g_latest_frame.ready.wait_for(lock, std::chrono::milliseconds(500), [&] {
                return !g_running || g_latest_frame.sequence != sent_sequence;
            });
            if (!g_running || g_latest_frame.sequence == sent_sequence) continue;
            jpeg = g_latest_frame.jpeg;
            sent_sequence = g_latest_frame.sequence;
        }
        if (jpeg.empty()) continue;

        const std::string part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
            std::to_string(jpeg.size()) + "\r\n\r\n";
        if (!sendText(client_fd, part) || !sendAll(client_fd, jpeg.data(), jpeg.size()) ||
            !sendText(client_fd, "\r\n")) {
            break;
        }
    }
    ::close(client_fd);
}

void httpServerThread(int port, std::vector<std::thread>& clients, std::mutex& clients_mutex) {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Cannot create HTTP socket." << std::endl;
        return;
    }
    int enabled = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        ::listen(server_fd, 4) < 0) {
        std::cerr << "Cannot listen on HTTP port " << port << "." << std::endl;
        ::close(server_fd);
        return;
    }

    std::cout << "HTTP 控制与图传已启动: http://<树莓派IP>:" << port << "/" << std::endl;
    while (g_running) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(server_fd, &read_set);
        timeval timeout{1, 0};
        const int ready = select(server_fd + 1, &read_set, nullptr, nullptr, &timeout);
        if (ready <= 0 || !FD_ISSET(server_fd, &read_set)) continue;

        const int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        timeval client_timeout{2, 0};
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
                   &client_timeout, sizeof(client_timeout));
        char request[1024] = {};
        const ssize_t received = ::recv(client_fd, request, sizeof(request) - 1, 0);
        const std::string path = received > 0 ? requestPath(request) : "/";

        if (path.rfind("/control", 0) == 0) {
            const bool valid = updateWebControl(path);
            const std::string response = valid
                ? "HTTP/1.1 204 No Content\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
                : "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n";
            sendText(client_fd, response);
            ::close(client_fd);
        } else if (path.rfind("/stream", 0) == 0) {
            std::lock_guard<std::mutex> lock(clients_mutex);
            clients.emplace_back(handleStreamClient, client_fd);
        } else {
            sendPage(client_fd);
            ::close(client_fd);
        }
    }
    ::close(server_fd);
}

void cameraThread(int camera_id) {
    cv::VideoCapture camera(camera_id);
    if (!camera.isOpened()) {
        std::cerr << "Cannot open camera " << camera_id << "." << std::endl;
        return;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, kCaptureWidth);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, kCaptureHeight);
    camera.set(cv::CAP_PROP_FPS, kStreamFps);
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

    const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, kJpegQuality};
    const auto frame_interval = std::chrono::milliseconds(1000 / kStreamFps);
    while (g_running) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "Camera frame read failed." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(kStreamWidth, kStreamHeight),
                   0.0, 0.0, cv::INTER_AREA);
        frame = std::move(resized);

        const int drive = g_drive_state.load();
        const int steering = g_steering_state.load();
        const char* drive_text = drive > 0 ? "FORWARD" : drive < 0 ? "REVERSE" : "STOP";
        const char* steering_text = steering < 0 ? "LEFT" : steering > 0 ? "RIGHT" : "CENTER";
        cv::putText(frame, std::string("Drive: ") + drive_text + "  Steering: " + steering_text +
                                "  PWM: " + std::to_string(g_motor_pwm.load()),
                    cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        std::vector<uchar> jpeg;
        if (cv::imencode(".jpg", frame, jpeg, jpeg_params)) {
            {
                std::lock_guard<std::mutex> lock(g_latest_frame.mutex);
                g_latest_frame.jpeg = std::move(jpeg);
                ++g_latest_frame.sequence;
            }
            g_latest_frame.ready.notify_all();
        }
        std::this_thread::sleep_for(frame_interval);
    }
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [http_port] [camera_id] [IN1_GPIO] [IN2_GPIO]\n"
              << "Default: port=8090, camera=0, no reverse direction GPIOs.\n"
              << "Example with H-bridge: " << program << " 8090 0 5 6" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    int port = kDefaultPort;
    int camera_id = kDefaultCamera;
    int direction_a_pin = -1;
    int direction_b_pin = -1;
    if (argc > 5 || (argc >= 2 && !parseInt(argv[1], port)) ||
        (argc >= 3 && !parseInt(argv[2], camera_id)) ||
        (argc >= 4 && !parseInt(argv[3], direction_a_pin)) ||
        (argc >= 5 && !parseInt(argv[4], direction_b_pin)) ||
        port < 1 || port > 65535) {
        printUsage(argv[0]);
        return 1;
    }

    // pigpio otherwise installs its own SIGINT handler and bypasses safe cleanup.
    if (gpioCfgSetInternals(PI_CFG_NOSIGHANDLER) < 0) {
        std::cerr << "Cannot disable pigpio signal handler." << std::endl;
        return 1;
    }
    if (gpioInitialise() < 0) {
        std::cerr << "pigpio initialization failed. Run with sudo and check GPIO access." << std::endl;
        return 1;
    }

    struct sigaction signal_action {};
    signal_action.sa_handler = onSignal;
    sigemptyset(&signal_action.sa_mask);
    signal_action.sa_flags = 0;
    if (sigaction(SIGINT, &signal_action, nullptr) < 0 ||
        sigaction(SIGTERM, &signal_action, nullptr) < 0) {
        std::cerr << "Cannot install signal handlers." << std::endl;
        gpioTerminate();
        return 1;
    }

    MotorDriver motor(kMotorPwmPin, direction_a_pin, direction_b_pin);
    SteeringServo servo;
    if (!motor.initialize() || !servo.initialize()) {
        std::cerr << "GPIO initialization failed." << std::endl;
        motor.stop();
        gpioTerminate();
        return 1;
    }
    servo.center();

    std::vector<std::thread> clients;
    std::mutex clients_mutex;
    std::thread camera(cameraThread, camera_id);
    std::thread http_server(httpServerThread, port, std::ref(clients), std::ref(clients_mutex));

    std::cout << "网页控制：浏览器打开图传地址后，点击页面并使用方向键或 W/A/S/D。" << std::endl;
    std::cout << "松开按键、浏览器失焦或网页心跳中断时，车辆立即停车并回正。" << std::endl;
    if (direction_a_pin < 0) {
        std::cout << "单路双向电调：默认前进 PWM=" << kMotorDrivePwm
                  << "，后退 PWM=" << kMotorReversePwm << "（网页可单独调整）。" << std::endl;
    }

    int motor_pwm = 0;
    int applied_web_drive = 0;
    int applied_web_steering = 0;
    int applied_web_pwm = kMotorDrivePwm;
    bool reverse_notice_printed = false;

    const auto stop_car = [&] {
        motor.stop();
        servo.center();
        motor_pwm = 0;
        g_motor_pwm = 0;
        g_drive_state = 0;
        g_steering_state = 0;
    };
    const auto drive_car = [&](int direction, int pwm) {
        const bool is_starting = g_drive_state.load() != direction || motor_pwm == 0;
        if (!is_starting) return true;

        motor_pwm = clampInt(pwm, 0, kMotorPwmRange);

        // 双击倒车保护：从前进直接切后退，电调会当作刹车；先回中位停顿再给后退。
        if (direction < 0 && direction_a_pin < 0 && g_drive_state.load() == 1) {
            motor.stop();  // 回中位 10000（刹车）
            std::this_thread::sleep_for(kReverseBrakePause);
        }

        if (!motor.drive(direction, motor_pwm)) {
            stop_car();
            if (!reverse_notice_printed) {
                std::cout << "未配置倒车方向 GPIO，下键已执行停车。" << std::endl;
                reverse_notice_printed = true;
            }
            return false;
        }
        g_motor_pwm = motor_pwm;
        g_drive_state = direction;
        return true;
    };
    const auto steer_car = [&](int steering) {
        servo.setDirection(steering);
        g_steering_state = steering;
    };

    while (g_running && !g_stop_requested) {
        const WebControlSnapshot web = currentWebControl();
        const int requested_drive = web.active ? web.drive : 0;
        const int requested_steering = web.active ? web.steering : 0;
        const int requested_pwm = requested_drive < 0 ? web.reverse_pwm : web.forward_pwm;

        if (requested_drive != applied_web_drive) {
            if (requested_drive == 0) {
                stop_car();
            } else {
                drive_car(requested_drive, requested_pwm);
            }
            applied_web_drive = requested_drive;
        } else if (requested_drive != 0 && requested_pwm != applied_web_pwm) {
            motor_pwm = clampInt(requested_pwm, 0, kMotorPwmRange);
            motor.setPwm(motor_pwm);
            g_motor_pwm = motor_pwm;
        }
        applied_web_pwm = requested_pwm;
        if (requested_steering != applied_web_steering) {
            if (requested_steering == 0) {
                servo.centerImmediately();
                g_steering_state = 0;
            } else {
                steer_car(requested_steering);
            }
            applied_web_steering = requested_steering;
        }

        servo.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    stop_car();
    servo.centerImmediately();
    g_running = false;
    g_latest_frame.ready.notify_all();
    if (camera.joinable()) camera.join();
    if (http_server.joinable()) http_server.join();
    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        for (std::thread& client : clients) {
            if (client.joinable()) client.join();
        }
    }
    gpioTerminate();
    return 0;
}
