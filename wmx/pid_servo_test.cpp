// 更新说明：增加舵机 PID 参数和角度范围测试，便于标定回中角度及转向响应。
// 更新日期：2026-09
#include <pigpio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

constexpr int kServoPin = 12;
constexpr int kServoFrequencyHz = 50;
constexpr int kServoCenterAngle = 85;
constexpr int kServoMinAngle = 60;
constexpr int kServoMaxAngle = 120;
constexpr int kMinPulseWidthUs = 500;
constexpr int kMaxPulseWidthUs = 2500;
constexpr double kFrameDtSeconds = 1.0 / 25.0;

std::atomic<bool> g_running(true);

void onSignal(int) {
    g_running = false;
}

struct PIDGains {
    double kp = 0.0;
    double ki = 0.0;
    double kd = 0.0;
};

enum class PIDSegment {
    kDeadband,
    kSmallError,
    kLargeError,
};

const char* segmentName(PIDSegment segment) {
    switch (segment) {
        case PIDSegment::kDeadband: return "deadband";
        case PIDSegment::kSmallError: return "3..10";
        case PIDSegment::kLargeError: return ">10";
    }
    return "unknown";
}

struct PIDResult {
    double control = 0.0;
    PIDSegment segment = PIDSegment::kDeadband;
};

class PiecewisePositionPID {
public:
    PiecewisePositionPID(PIDGains small_left_gains,
                         PIDGains small_right_gains,
                         PIDGains large_left_gains,
                         PIDGains large_right_gains,
                         double deadband, double split_error,
                         double integral_limit = 200.0)
        : small_left_gains_(small_left_gains),
          small_right_gains_(small_right_gains),
          large_left_gains_(large_left_gains),
          large_right_gains_(large_right_gains),
          deadband_(std::abs(deadband)),
          split_error_(std::max(std::abs(deadband), std::abs(split_error))),
          integral_limit_(std::abs(integral_limit)) {}

    PIDResult update(double error, double dt) {
        if (std::abs(error) <= deadband_) {
            reset();
            return {0.0, PIDSegment::kDeadband};
        }

        dt = std::clamp(dt, 0.01, 0.20);
        const bool small_error = std::abs(error) <= split_error_;
        const bool left_turn = error < 0.0;
        const PIDGains& gains = small_error
            ? (left_turn ? small_left_gains_ : small_right_gains_)
            : (left_turn ? large_left_gains_ : large_right_gains_);
        const double derivative = initialized_
            ? (error - last_error_) / dt : 0.0;
        integral_ = std::clamp(integral_ + error * dt,
                               -integral_limit_, integral_limit_);

        last_error_ = error;
        initialized_ = true;
        // No output limit: SteeringServo still clamps to 60..120 degrees.
        return {
            gains.kp * error + gains.ki * integral_ + gains.kd * derivative,
            small_error ? PIDSegment::kSmallError : PIDSegment::kLargeError,
        };
    }

    void reset() {
        integral_ = 0.0;
        last_error_ = 0.0;
        initialized_ = false;
    }

private:
    PIDGains small_left_gains_;
    PIDGains small_right_gains_;
    PIDGains large_left_gains_;
    PIDGains large_right_gains_;
    double deadband_;
    double split_error_;
    double integral_limit_;
    double integral_ = 0.0;
    double last_error_ = 0.0;
    bool initialized_ = false;
};

class SteeringServo {
public:
    bool initialize() {
        if (gpioSetMode(kServoPin, PI_OUTPUT) < 0) return false;
        if (gpioSetPWMfrequency(kServoPin, kServoFrequencyHz) < 0) return false;
        center();
        return true;
    }

    int setFromControl(double control) {
        const int angle = std::clamp(
            static_cast<int>(std::lround(kServoCenterAngle - control)),
            kServoMinAngle, kServoMaxAngle);
        gpioServo(kServoPin, angleToPulseWidth(angle));
        return angle;
    }

    void center() {
        gpioServo(kServoPin, angleToPulseWidth(kServoCenterAngle));
    }

private:
    static int angleToPulseWidth(int angle) {
        return kMinPulseWidthUs +
               (angle * (kMaxPulseWidthUs - kMinPulseWidthUs)) / 180;
    }
};

void printHelp() {
    std::cout << "\nPID servo test, motor is not controlled.\n"
              << "Current setup: center=85 deg, limits=60..120 deg, 25 FPS.\n"
              << "PID: deadband <=3px; left 3..10: P=0.22 D=0.0016; "
              << "right 3..10: P=0.30 D=0.0022; left >10: P=0.28 D=0.0022; "
              << "right >10: P=0.38 D=0.0030.\n"
              << "Input: <error> [frames], for example: 8 1 or -20 10\n"
              << "Commands: reset, help, q\n\n";
}

}  // namespace

int main() {
    if (gpioCfgSetInternals(PI_CFG_NOSIGHANDLER) < 0 || gpioInitialise() < 0) {
        std::cerr << "pigpio initialization failed. Run with sudo and check GPIO12." << std::endl;
        return 1;
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    SteeringServo servo;
    if (!servo.initialize()) {
        std::cerr << "Servo GPIO12 initialization failed." << std::endl;
        gpioTerminate();
        return 1;
    }

    PiecewisePositionPID pid(
        PIDGains{0.22, 0.00, 0.0016},  // 3..10px, left
        PIDGains{0.30, 0.00, 0.0022},  // 3..10px, right
        PIDGains{0.28, 0.00, 0.0022},  // >10px, left
        PIDGains{0.38, 0.00, 0.0030},  // >10px, right
        3.0, 10.0);

    printHelp();
    std::string line;
    while (g_running && std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "q" || line == "Q" || line == "quit") break;
        if (line == "help" || line == "h") {
            printHelp();
            continue;
        }
        if (line == "reset" || line == "r") {
            pid.reset();
            servo.center();
            std::cout << "PID reset, servo=85 deg, pulse=1444us" << std::endl;
            continue;
        }

        std::istringstream input(line);
        double error = 0.0;
        int frames = 1;
        if (!(input >> error) || (input >> frames && frames < 1)) {
            std::cout << "Invalid input. Example: 8 1, -20 10, reset, or q" << std::endl;
            continue;
        }
        frames = std::clamp(frames, 1, 250);

        for (int frame = 1; frame <= frames && g_running; ++frame) {
            const PIDResult result = pid.update(error, kFrameDtSeconds);
            const int angle = servo.setFromControl(result.control);
            const int pulse = kMinPulseWidthUs +
                (angle * (kMaxPulseWidthUs - kMinPulseWidthUs)) / 180;
            const char* direction = error < 0.0 ? "left" :
                                    error > 0.0 ? "right" : "center";
            std::cout << std::fixed << std::setprecision(2)
                      << "frame=" << frame
                      << " error=" << error
                      << " direction=" << direction
                      << " segment=" << segmentName(result.segment)
                      << " control=" << result.control
                      << " angle=" << angle
                      << " pulse=" << pulse << "us" << std::endl;
            if (frame < frames) {
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }
        }
    }

    servo.center();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    gpioTerminate();
    return 0;
}
