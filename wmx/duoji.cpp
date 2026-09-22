// 更新说明：舵机 GPIO 改为 BCM GPIO12，并改用 510~2510us 脉宽映射角度；扫描范围调整为 60~120°。
// 更新日期：2026-09
#include <pigpio.h>
#include <iostream>
#include <unistd.h>

using namespace std;

const int SERVO_PIN = 12;

static int angleToPulseWidth(int angle) {
    // gpioServo 的参数单位是微秒，常见舵机范围为 500~2500us。
    return 510 + (angle * 2000) / 180;
}

static void setServoAngle(int angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    gpioServo(SERVO_PIN, angleToPulseWidth(angle));
}

static void servoTestSequence() {
    cout << "=== 舵机测试开始 ===" << endl;

    cout << "1. 初始化到 90°" << endl;
    setServoAngle(90);
    sleep(1);

    cout << "2. 左转扫描 60° -> 120°" << endl;
    for (int angle = 60; angle <= 120; angle += 10) {
        cout << "angle = " << angle << "°" << endl;
        setServoAngle(angle);
        sleep(1);
    }

    cout << "3. 右转扫描 120° -> 60°" << endl;
    for (int angle = 120; angle >= 60; angle -= 10) {
        cout << "angle = " << angle << "°" << endl;
        setServoAngle(angle);
        sleep(1);
    }

    cout << "4. 回中" << endl;
    setServoAngle(90);
    sleep(2);

    cout << "=== 舵机测试结束 ===" << endl;
}

int main() {
    if (gpioInitialise() < 0) {
        cerr << "gpio 初始化失败，确认是否已运行 pigpiod" << endl;
        return 1;
    }

    gpioSetMode(SERVO_PIN, PI_OUTPUT);
    gpioSetPWMfrequency(SERVO_PIN, 50);

    setServoAngle(90);
    sleep(1);

    servoTestSequence();

    gpioTerminate();
    return 0;
}
