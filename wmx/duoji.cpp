#include <pigpio.h>
#include <iostream>
#include <unistd.h>

using namespace std;

const int SERVO_PIN = 17;

static double angleToDutyCycle(int angle) {
    // 50Hz 伺服控制，通常 0.5ms~2.5ms 对应 0~180°
    // 这里按常见写法映射到 0~180°
    return 2.5 + (angle / 180.0) * 10.0;
}

static void setServoAngle(int angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    double duty = angleToDutyCycle(angle);
    gpioServo(SERVO_PIN, static_cast<int>(duty * 1000.0));
}

static void servoTestSequence() {
    cout << "=== 舵机测试开始 ===" << endl;

    cout << "1. 初始化到 90°" << endl;
    setServoAngle(90);
    sleep(2);

    cout << "2. 左转扫描 60° -> 140°" << endl;
    for (int angle = 60; angle <= 140; angle += 10) {
        cout << "angle = " << angle << "°" << endl;
        setServoAngle(angle);
        sleep(1);
    }

    cout << "3. 右转扫描 140° -> 60°" << endl;
    for (int angle = 140; angle >= 60; angle -= 10) {
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
