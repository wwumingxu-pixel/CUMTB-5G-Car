// 更新说明：电机由固定流程改为命令行手动 PWM 测试，可输入 PWM 值测试停止、前进和后退；退出自动回到 10000 中位。
// 更新日期：2026-09
#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std;

// 实测确认：中位 10000 停止；>10000 前进；<10000 后退（9500 开始后退）。
const int MOTOR_PIN = 13;
const int PWM_RANGE = 40000;
const int PWM_FREQ = 200;

const int STOP_DUTY = 10000;     // 中位停止
const int FORWARD_DUTY = 11000;  // 前进
const int REVERSE_DUTY = 9000;   // 后退（9500 开始，取 9000 稳定后退）

static volatile sig_atomic_t g_stop = 0;

static void onSignal(int) {
    g_stop = 1;
}

static void setDuty(int duty) {
    if (duty < 0) duty = 0;
    if (duty > PWM_RANGE) duty = PWM_RANGE;
    gpioPWM(MOTOR_PIN, duty);
    cout << "duty=" << duty << endl;
}

static void sleepFor(int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!g_stop && std::chrono::steady_clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

static void manualControl() {
    cout << "=== 电机手动调试 ===" << endl;
    cout << "中位 " << STOP_DUTY << " 停止；前进参考 " << FORWARD_DUTY
         << "；后退参考 " << REVERSE_DUTY << "（9500 开始后退）。" << endl;
    cout << "输入 0~40000 设 duty；q 回中位退出。" << endl;

    string input;
    while (!g_stop) {
        cout << "> " << flush;
        if (!(cin >> input)) break;
        if (input == "q" || input == "Q") break;
        try {
            size_t used = 0;
            const int value = stoi(input, &used);
            if (used != input.size()) throw invalid_argument("extra characters");
            setDuty(value);
        } catch (...) {
            cout << "输入无效，请输入 0~40000 的整数或 q。" << endl;
        }
    }

    setDuty(STOP_DUTY);
    cout << "已回中位 " << STOP_DUTY << "。" << endl;
}

int main() {
    if (gpioCfgSetInternals(PI_CFG_NOSIGHANDLER) < 0) {
        cerr << "无法禁用 pigpio 信号处理。" << endl;
        return 1;
    }
    if (gpioInitialise() < 0) {
        cerr << "gpio 初始化失败。请使用 sudo 运行，并确认 GPIO13 未被其他程序占用。" << endl;
        return 1;
    }

    struct sigaction sa {};
    sa.sa_handler = onSignal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    if (gpioSetMode(MOTOR_PIN, PI_OUTPUT) < 0) {
        cerr << "GPIO" << MOTOR_PIN << " cannot be configured as output" << endl;
        gpioTerminate();
        return 1;
    }

    // 顺序：先设频率，再设范围（此顺序下 duty 才有 0~40000 区分度，10000 为中位）。
    const int freq = gpioSetPWMfrequency(MOTOR_PIN, PWM_FREQ);
    const int range = gpioSetPWMrange(MOTOR_PIN, PWM_RANGE);
    cout << "Motor GPIO" << MOTOR_PIN << " range=" << range << ", freq=" << freq << " Hz" << endl;

    setDuty(STOP_DUTY);  // 先回中位
    sleepFor(2000);

    manualControl();

    gpioTerminate();
    return 0;
}
