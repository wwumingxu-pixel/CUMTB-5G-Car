#include <pigpio.h>
#include <iostream>
#include <unistd.h>

using namespace std;

// 这个项目里用的是单路 PWM 控制电机
// 如果你们实际接线是 L298N / TB6612FNG 这类驱动板，
// 需要按自己的 IN1/IN2/ENA 连接方式来改。
const int MOTOR_PIN = 13;
const int PWM_RANGE = 40000;
const int PWM_FREQ = 200;

static void setMotorPwm(int value) {
    if (value < 0) value = 0;
    if (value > PWM_RANGE) value = PWM_RANGE;
    gpioPWM(MOTOR_PIN, value);
}

static void motorTestSequence() {
    cout << "=== 马达测试开始 ===" << endl;
    cout << "1. 停止 2 秒" << endl;
    setMotorPwm(0);
    sleep(2);

    cout << "2. 从 10000 开始缓慢加速，步进 1000" << endl;
    for (int v = 10000; v <= 16000; v += 1000) {
        cout << "PWM = " << v << endl;
        setMotorPwm(v);
        usleep(600000); // 600ms
    }

    cout << "3. 保持高速 16000" << endl;
    setMotorPwm(16000);
    sleep(2);

    cout << "4. 缓慢减速" << endl;
    for (int v = 16000; v >= 10000; v -= 1000) {
        cout << "PWM = " << v << endl;
        setMotorPwm(v);
        usleep(600000); // 600ms
    }

    cout << "5. 停止" << endl;
    setMotorPwm(0);
    sleep(1);

    cout << "=== 马达测试结束 ===" << endl;
}

int main() {
    if (gpioInitialise() < 0) {
        cerr << "gpio 初始化失败，确认是否已运行 pigpiod" << endl;
        return 1;
    }

    gpioSetMode(MOTOR_PIN, PI_OUTPUT);
    gpioSetPWMrange(MOTOR_PIN, PWM_RANGE);
    gpioSetPWMfrequency(MOTOR_PIN, PWM_FREQ);

    // 初始化时先停下
    gpioPWM(MOTOR_PIN, 0);
    sleep(1);

    motorTestSequence();

    gpioTerminate();
    return 0;
}
