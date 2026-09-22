// 更新说明：增加可配置 PWM 扫描区间、步进和停留时间，用于现场标定停止、前进、后退对应的 PWM。
// 更新日期：2026-09
#include <pigpio.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

constexpr int kMotorPin = 13;       // BCM GPIO13, physical pin 33
constexpr int kPwmRange = 40000;
constexpr int kPwmFreq = 200;

volatile sig_atomic_t g_stop = 0;

void onSignal(int) {
    g_stop = 1;
}

bool parseInt(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [start] [end] [step] [dwell_ms]\n"
              << "Example: " << program << " 0 40000 500 800\n"
              << "Scans GPIO13 PWM from start to end. Ctrl+C stops immediately." << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    int start = 0;
    int end = 40000;
    int step = 500;
    int dwell_ms = 800;

    if (argc > 5 ||
        (argc >= 2 && !parseInt(argv[1], start)) ||
        (argc >= 3 && !parseInt(argv[2], end)) ||
        (argc >= 4 && !parseInt(argv[3], step)) ||
        (argc >= 5 && !parseInt(argv[4], dwell_ms)) ||
        start < 0 || end > kPwmRange || start > end || step <= 0 || dwell_ms < 50) {
        printUsage(argv[0]);
        return 1;
    }

    if (gpioCfgSetInternals(PI_CFG_NOSIGHANDLER) < 0 || gpioInitialise() < 0) {
        std::cerr << "pigpio initialization failed. Run with sudo." << std::endl;
        return 1;
    }

    struct sigaction signal_action {};
    signal_action.sa_handler = onSignal;
    sigemptyset(&signal_action.sa_mask);
    sigaction(SIGINT, &signal_action, nullptr);
    sigaction(SIGTERM, &signal_action, nullptr);

    if (gpioSetMode(kMotorPin, PI_OUTPUT) < 0) {
        std::cerr << "GPIO13 configuration failed." << std::endl;
        gpioTerminate();
        return 1;
    }
    const int actual_range = gpioSetPWMrange(kMotorPin, kPwmRange);
    const int actual_frequency = gpioSetPWMfrequency(kMotorPin, kPwmFreq);
    std::cout << "Motor BCM GPIO=" << kMotorPin << ", range=" << actual_range
              << ", frequency=" << actual_frequency << " Hz" << std::endl;
    std::cout << "Scan " << start << " -> " << end << ", step=" << step
              << ", dwell=" << dwell_ms << " ms" << std::endl;
    std::cout << "*** 请先把车辆架空（驱动轮离地）再开始 ***" << std::endl;
    std::cout << "观察每个 PWM 值对应的方向：停 / 前进 / 后退，记录分界点。" << std::endl;

    for (int pwm = start; pwm <= end && !g_stop; pwm += step) {
        gpioPWM(kMotorPin, pwm);
        std::cout << "PWM=" << pwm << "   (duty " << (pwm * 100.0 / kPwmRange) << "%)" << std::endl;

        // 停留期间可随时 Ctrl+C 提前停车。
        const auto finish = std::chrono::steady_clock::now() + std::chrono::milliseconds(dwell_ms);
        while (!g_stop && std::chrono::steady_clock::now() < finish) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    gpioPWM(kMotorPin, 0);
    gpioTerminate();
    std::cout << "Scan finished. Motor stopped." << std::endl;
    return 0;
}
