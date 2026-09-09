#include <iostream>
#include <pigpio.h>
#include <cstdlib> // 用于 system 调用
#include <thread>  // 用于 sleep
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <chrono>
#include <numeric> 

#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include <atomic>
#include <sys/select.h>  // 用于 select、fd_set
#include <sys/time.h>    // 用于 timeval
#include <time.h>        // 用于 timespec、clock_gettime
#include <errno.h>       // 用于 ETIMEDOUT
#include <signal.h>      // 用于 signal、SIGINT
// 补充系统头文件，用于目录操作
#include <sys/types.h>   // 定义基本系统类型（如mode_t）
#include <sys/stat.h>    // 定义struct stat和mkdir函数

// 新增：图像保存相关常量（可放在main函数外或开头）
#define IMAGE_SAVE_DIR "/home/pi/5G_car/photo_2025/"  // 保存目录（可自定义）
#define IMAGE_SAVE_INTERVAL_SEC 30        // 保存间隔：30秒

// 巡线调试宏
#define LINE_DEBUG
#define DRAW_MID

using namespace std;
using namespace cv;

class Config {
public:
    // 全局计数变量，程序每运行一次+1
    static int count_compare;

    // 赛道颜色提取
    static std::array<int, 3> lower_red1; // [0, 50, 46]
    static const std::array<int, 3> upper_red1; // [10, 255, 255]
    static std::array<int, 3> lower_red2; // [100, 50, 46]
    static const std::array<int, 3> upper_red2; // [179, 255, 255]
    static const std::array<int, 3> lower_yellow;
    static const std::array<int, 3> upper_yellow;

    // 锥桶颜色提取
    static const std::array<int, 3> lower_blue; // [100, 79, 124]
    static const std::array<int, 3> upper_blue; // [140, 255, 255]
    static const int blue_mask_down = 120; // 锥桶图像蒙版最顶行
    static const int blue_mask_up = 180; // 锥桶图像蒙版最底行
    static const int min_width = 5; // 锥桶识别大小限制
    static const int min_height = 5;
    static const int max_width = 300;
    static const int max_height = 300;

    // 巡线参数
    static const int x_middle = 320; // 理论中线
    static const int wide_scan = 10; // x方向每次取一行的5个像素点
    static const int wide_need = 4; // 5个像素点≥2，则检测到边线
    static const int up_scan_wide = 90; // 从图像的130行开始巡线
    static const int down_scan_wide = 170; // 从图像的170行结束巡线
    static const int y_scan_wide = 10; // y方向5行检测一次边线
    static std::array<uint16_t, 10> draw_middle_line_x; // 存储检测到的中线点
    static std::array<uint16_t, 10> draw_left_line_x; // 存储检测到的左边线点
    static std::array<uint16_t, 10> draw_right_line_x; // 存储检测到的右边线点
    static int error; // 误差
    static int last_error; // 记录上一次误差
    static const double kp;
    static const double kd;
    static const int all_count; // 共遍历all_count行
    // 锥桶检测参数
    static int turn_detect_res; // 锥桶检测结果.1左转，2右转
    static int cone_x; // 锥桶检测中心x坐标
    static int cone_y; // 锥桶检测中心y坐标
    static int cone_count; // 过一个锥桶则+1
    static int count; // 每张图片出现锥桶则+1
    static int left_lost_time; // 记录左丢线数量，以此判断第一个锥桶从左还是从右绕
    static int right_lost_time;
    
    //转向检测参数
    static int turn_angle; // 转向固定转角值 servo_middle - turn_angle
    static int turn_line; // 转弯处理中检测第turn_line行
    static int turn_time; // 记录转向用时

    //AB区域停车参数
    static int AB_x;
    static int AB_y;

    //黄色锥桶连线计算
    static float yellow_cone_k;
    static float yellow_cone_b;
};

// 标志位
class State {
public:
    static bool sidewalk; // 斑马线标志，过斑马线置为False
    static bool sidewalk_end_flag;

    static int cone_start; // 锥桶开始标志
    static int cone_detect_state_1; // 记录出现过出现锥桶的情况
    static int cone_detect_state_2; // 记录没有锥桶的情况
    static int left_or_right_turn; // 碰到第一个锥桶时选择是左转还是右转。1为左转，2为右转
    
    static bool turn_end_flag; // 转向结束标志  


    static int yellow_cone_start;//黄色锥桶第一次返回开始标志
    static int yellow_cone_end;//黄色锥桶第一次返回结束标志
    static int yellow_cone_start_2;//黄色锥桶第二次检测开始标志
    static int yellow_cone_end_2;//黄色锥桶第二次检测结束标志

    // 换道循环内判断使用
    static int left_right_exit_flag; 
    static int left_exit_flag; 
    static int right_exit_flag; 
    
    //停车检测标志位
    static int have_one_blue; // 检测到第一个蓝色区域(A || B)
    static int have_two_blue; // 检测到两个蓝色区域
    static int AB_flag;// AB开始标志
    static int AB_flag2;// AB开始标志
};

// 车道变换状态结构体：存储与车道识别相关的所有信息
struct LaneChangeState {
    int direction;                 // 变道方向：0-未知，1-左变道，2-右变道
    vector<Point2f> blue_cones;    // 蓝色锥桶的坐标集合
    vector<Point2f> yellow_cones;  // 黄色锥桶的坐标集合
    vector<Point2f> guide_path;    // 生成的引导路径点集合
};

// 初始化静态成员
int Config::count_compare = 0;
std::array<int, 3> Config::lower_red1 = {0, 75, 70};
const std::array<int, 3> Config::upper_red1 = {34, 255, 255};
std::array<int, 3> Config::lower_red2 = {160, 75, 70};
const std::array<int, 3> Config::upper_red2 = {179, 255, 255};

const std::array<int, 3> Config::lower_yellow = {17, 56, 50};
const std::array<int, 3> Config::upper_yellow = {46, 255, 255};

const std::array<int, 3> Config::lower_blue = {100, 37, 100};
const std::array<int, 3> Config::upper_blue = {140, 255, 255};
std::array<uint16_t, 10> Config::draw_middle_line_x = {};
std::array<uint16_t, 10> Config::draw_left_line_x = {};
std::array<uint16_t, 10> Config::draw_right_line_x = {};
int Config::error = 0;
int Config::last_error = 0;
const double Config::kp = 0.3; // 0.6
const double Config::kd = 2;
const int Config::all_count = (Config::down_scan_wide - Config::up_scan_wide) / Config::y_scan_wide;

int Config::cone_x = 0;
int Config::cone_y = 0;
int Config::cone_count = 1;
int Config::count = 0;
int Config::left_lost_time = 0;
int Config::right_lost_time = 0;


int Config::turn_time = 0;
int Config::turn_detect_res = 1;

int Config::AB_x = 0;
int Config::AB_y = 0;

int turn_line = 5;

// 正常模式：按照完整流程执行
bool State::sidewalk = false;
bool State::sidewalk_end_flag = false;
bool State::turn_end_flag = false;//原false
int State::cone_start = 0;
int State::cone_detect_state_1 = 0;
int State::cone_detect_state_2 = 1;
int State::left_or_right_turn = 0;
int State::left_right_exit_flag = 0; 
int State::left_exit_flag = 0; 
int State::right_exit_flag = 0; 
int State::have_one_blue = 0;
int State::have_two_blue = 0;
int State::AB_flag = 0;
int State::AB_flag2 = 0;
int State::yellow_cone_start = 0;
int State::yellow_cone_end = 0;
int State::yellow_cone_start_2 = 0;
int State::yellow_cone_end_2 = 0;

// 2025.10.7 AB检测参数
const int MIN_AREA_THRESHOLD = 200;     // 最小面积阈值
const float ASPECT_RATIO_MAX = 10.0;   // 最大宽高比
const float ASPECT_RATIO_MIN = 0.0;    // 最小宽高比
const double AB_AREA_RATIO_THRESHOLD = 0.85;  // AB判断阈值（contour_area/hull_area > 0.85为B，否则为A）
int A_result = 0;
int B_result = 0;
// 新版AB检测投票相关变量
int AB_detect_results[3] = {0};   // 记录每次检测到的结果（1=A, 2=B）
int AB_detect_count = 0;          // 已经记录的有效检测次数（最多3次）
int final_result = 0 ;
int AB_frame_count = 0;           // 开始AB检测后的帧计数，用于超时
const int AB_MAX_FRAMES = 60;     // 在多少帧内如果还不到3次，就用第二次结果（可根据速度调）
//巡线参数
const double SCAN_TOP_RATIO = 0.5;  // 对应xunxian的扫描范围比例
const double SCAN_BOTTOM_RATIO = 0.7;

// 斑马线检测参数 - 针对320x180分辨率调整
const int MAX_AREA_THRESHOLD = 1500;    // 最大面积
const float MIN_ASPECT_RATIO = 1;     // 最小宽高比
const float MAX_ASPECT_RATIO = 3.0;     // 最大宽高比
const float COLINEAR_THRESHOLD = 30.0;   // 共线阈值（与zebra.cpp保持一致）
const float ANGLE_THRESHOLD = 25.0;     // 角度阈值

float Config::yellow_cone_k=0;
float Config::yellow_cone_b=0;

// 锥桶检测通用参数
const int MIN_CONTOUR_AREA = 50;        // 最小轮廓面积
const int MAX_CONTOUR_AREA = 1000;      // 最大轮廓面积
const float MIN_CONE_ASPECT_RATIO = 1.0f;    // 最小宽高比
const float MAX_CONE_ASPECT_RATIO = 2.0f;    // 最大宽高比
const int MIN_CONES_COUNT = 3;          // 触发检测的最小黄色锥桶数量
const int MIN_BLUE_CONES_COUNT = 1;     // 触发蓝色锥桶检测的最小数量
bool zhuitong_had_blue = false;

int Weight_th[96] = { 
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8,
     7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1  }; 

int mid_line = 160, left_up_line = 0, right_up_line = 0, up_line_count = 0;

// 电机和舵机gpio引脚
const int motor_pin = 13;
const int servo_pin = 12;

// 舵机转向参数
const int servo_middle = 117; //115,原120
const int servo_min = 77;  // 最右边
const int servo_max = 153;  // 最左边
int angle_outmax = 8;  // 9
int angle_outmin = -8; // -9
const int K_TO_ANGLE = 20;      // k到角度的映射系数（初始值，需校准）
const int DEAD_ZONE = 0.1;         // k的死区（小于此值不转向，避免抖动）
const int SERVO_MIN = 40;    //角度最小值（对应右转）
const int SERVO_MAX = 140;   //角度最大值（对应左转）

const int ROI_Y_OFFSET = 90;

// 视频帧长宽
const int frame_width = 320;
const int frame_height = 180;
int asd = 0;

// 摄像头配置参数
const int CAMERA_PORT = 0;              // 摄像头端口
const int CAMERA_EXPOSURE = -13;        // 摄像头曝光度（最低值）
const int CAMERA_WIDTH = 320;           // 图像宽度
const int CAMERA_HEIGHT = 180;          // 图像高度

// 用于标记主程序的运行状态
std::atomic<bool> running(true);

// 双缓冲摄像头类 - 提高图像读取稳定性和效率
class DoubleBufferCamera {
public:
    DoubleBufferCamera(int src = 0) : running_flag(true), frontBuffer(0), released(false) {
        cap.open(src);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开摄像头");
        }
        // 设置摄像头参数
        cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);
        cap.set(CAP_PROP_EXPOSURE, CAMERA_EXPOSURE);
        cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
        cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
        
        // 启动线程异步读取帧
        try {
            captureThread = std::thread(&DoubleBufferCamera::update, this);
        } catch (const std::exception& e) {
            cap.release();  // 线程启动失败时清理摄像头资源
            throw std::runtime_error(std::string("线程启动失败: ") + e.what());
        }
    }

    ~DoubleBufferCamera() {
        release();
    }

    bool read(cv::Mat& frame) {
        int fb = frontBuffer.load(std::memory_order_acquire);
        if (buffers[fb].empty()) {
            return false;
        }
        buffers[fb].copyTo(frame);
        return !frame.empty();
    }

    void release() {
        // 幂等性保护：防止多次调用导致重复释放
        bool expected = false;
        if (!released.compare_exchange_strong(expected, true)) {
            return;  // 已经释放过，直接返回
        }
        
        running_flag = false;
        if (captureThread.joinable()) {
            captureThread.join();
        }
        if (cap.isOpened()) {
            cap.release();
        }
        
        // 清空缓冲区释放内存
        buffers[0].release();
        buffers[1].release();
    }
    
    bool isOpened() const {
        return cap.isOpened() && !released.load();
    }

private:
    VideoCapture cap;
    std::atomic<bool> running_flag;      // 线程运行标志
    std::atomic<bool> released;          // 资源释放标志（防止重复释放）
    std::atomic<int> frontBuffer;        // 前缓冲区索引（0或1）
    cv::Mat buffers[2];                  // 双缓冲区
    std::thread captureThread;           // 异步读取线程

    void update() {
        while (running_flag.load(std::memory_order_relaxed)) {  // 仅依赖内部标志
            cv::Mat frame;
            if (cap.read(frame)) {
                // 确保图像尺寸正确，如果不正确则调整为640x480
                if (frame.cols != 640 || frame.rows != 480) {
                    resize(frame, frame, Size(640, 480));
                }
                
                int fb = frontBuffer.load(std::memory_order_relaxed);
                int backBuffer = 1 - fb;
                frame.copyTo(buffers[backBuffer]);
                frontBuffer.store(backBuffer, std::memory_order_release);
            } else {
                // 摄像头读取失败，短暂休眠后重试
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
};

// 全局双缓冲摄像头指针（用于信号处理器中断开）
std::atomic<DoubleBufferCamera*> g_cap_ptr(nullptr);

// Ctrl+C 信号处理函数
void signalHandler(int signum) {
    if (signum == SIGINT) {
        std::cout << "\n\n收到 Ctrl+C 信号，正在安全退出..." << std::endl;
        running = false;
        
        // 关键修复：强制释放双缓冲摄像头以中断阻塞
        DoubleBufferCamera* cap = g_cap_ptr.load();
        if (cap != nullptr && cap->isOpened()) {
            std::cout << "[信号处理] 强制释放摄像头以中断阻塞..." << std::endl;
            cap->release();
        }
    }
}

// 输入监听线程的函数（修复：添加非阻塞输入检测）
void* listener(void* arg) {
    char buffer[256];
    while (running) {
        // 使用非阻塞方式检查输入
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms超时
        
        int ret = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
        if (ret > 0 && FD_ISSET(STDIN_FILENO, &readfds)) {
            if (fgets(buffer, sizeof(buffer), stdin)) {
                // 去掉换行符
                buffer[strcspn(buffer, "\n")] = 0;
                std::cout << "Received input: " << buffer << std::endl;
                
                if (strcmp(buffer, "s") == 0) {
                    State::sidewalk = true;
                    State::sidewalk_end_flag = true;
                }
                else if (strcmp(buffer, "b") == 0) {
                    State::turn_end_flag = true;
                    State::sidewalk_end_flag = true;
                    State::yellow_cone_end=1;
                }
                else if (strcmp(buffer, "a") == 0) {
                    State::turn_end_flag = true;
                    State::sidewalk_end_flag = true;
                }
                else if (strcmp(buffer, "q") == 0 || strcmp(buffer, "quit") == 0) {
                    std::cout << "收到退出指令，正在停止..." << std::endl;
                    running = false;
                }
                else {
                    running = false;
                }
            }
        }
        // 让出CPU，避免100%占用
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "监听线程已退出" << std::endl;
    return nullptr;
}

// 发车等待（修复：添加running检测和超时机制，使用双缓冲摄像头）
void waiting(DoubleBufferCamera& cap) {
    int timeout_counter = 0;
    const int max_wait_frames = 3000; // 最多等待3000帧（约100秒@30fps）
    
    while (running && timeout_counter < max_wait_frames) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            timeout_counter++;
            continue;
        }
        if (frame.empty()) {
            timeout_counter++;
            continue;
        }
        
        // 调整图像大小（关键修复：直接resize到frame，避免后续使用错误的图像）
        resize(frame, frame, Size(320, 180));
        
        // 添加调试信息
        if (timeout_counter % 30 == 0) {
            std::cout << "图像尺寸: " << frame.cols << "x" << frame.rows << std::endl;
        }
        
        // 转换到 HSV 颜色空间
        Mat frame_hsv;
        cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
        
        // 生成蓝色掩码
        Mat blue_mask;
        inRange(frame_hsv, Config::lower_blue, Config::upper_blue, blue_mask);
        
        // 添加调试信息
        int blue_pixels = cv::countNonZero(blue_mask);
        int total_pixels = frame.cols * frame.rows;
        double blue_percentage = (double)blue_pixels / total_pixels * 100;
        
        if (timeout_counter % 30 == 0) {
            std::cout << "蓝色像素: " << blue_pixels << "/" << total_pixels 
                      << " (" << blue_percentage << "%)" << std::endl;
        }
        
        // 蓝色小于50%启动
        if (blue_pixels < (total_pixels * 0.5)) {
            cout << "启动条件满足，蓝色区域小于50%" << endl;
            break;
        }
        
        if (timeout_counter % 30 == 0) {
            cout << "等待中... 蓝色区域仍大于50% (" << timeout_counter << "/" << max_wait_frames << ")" << endl;
        }
        
        timeout_counter++;
        
        // 检查退出按键
        if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q') {
            running = false;
            break;
        }
    }
    
    if (timeout_counter >= max_wait_frames) {
        std::cout << "等待超时，强制启动" << std::endl;
    }
    if (!running) {
        std::cout << "等待过程中收到退出指令" << std::endl;
    }
}

void gammaCorrection(const Mat& input, Mat& output) {
    // CV_Assert(gamma > 0); // 注释掉这行，因为gamma是局部变量
    // 将图像转换为灰度图像以计算平均亮度
    cv::Mat gray_image;
    cv::cvtColor(input, gray_image, cv::COLOR_BGR2GRAY);
    double mean_intensity = cv::mean(gray_image)[0];

    // 根据平均亮度选择伽马值
    // 光照强选80，其他选100即可
    double gamma = 0.5 + mean_intensity / 120.0; // 120 100 80  
    if (gamma > 2) gamma = 2;
    else if (gamma < 0.5) gamma = 0.5;
    // cout << gamma << endl;

    Mat lookupTable(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        lookupTable.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    LUT(input, lookupTable, output);
}

vector<cv::Point> extendLine(const cv::Point pt, double slope, double b, const cv::Size imageSize) {
    double x_intersect, y_intersect;
    vector<cv::Point> pts;
    x_intersect = (0 - b) / slope;
    if (x_intersect >= 0 && x_intersect < imageSize.width) {
        pts.push_back( cv::Point(x_intersect, 0) );
    }

    x_intersect = (imageSize.height - b) / slope;
    if (x_intersect >= 0 && x_intersect < imageSize.width) {
        pts.push_back( cv::Point(x_intersect, imageSize.height) );
    }

    y_intersect = slope * 0 + b;
    if (y_intersect >= 0 && y_intersect < imageSize.height) {
        pts.push_back( cv::Point(0, y_intersect) );
    }

    y_intersect = slope * imageSize.width + b;
    if (y_intersect >= 0 && y_intersect < imageSize.height) {
        pts.push_back( cv::Point(imageSize.width, y_intersect) );
    }
    if (pts.size()) return pts;
    return { };
}

// 车道线筛选函数
void filterLaneLines(Mat& binary_image) {
    vector<vector<Point>> contours;
    findContours(binary_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat lane_mask = Mat::zeros(binary_image.size(), CV_8UC1);
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        
        if (area < 30 || area > 5000) continue;
        
        RotatedRect min_rect = minAreaRect(contours[i]);
        Size2f rect_size = min_rect.size;
        float width = min(rect_size.width, rect_size.height);
        float height = max(rect_size.width, rect_size.height);
        
        float aspect_ratio = height / width;
        
        float angle = min_rect.angle;
        if (width > height) {
            angle += 90;
        }
        if (angle < 0) angle += 180;
        if (angle > 180) angle -= 180;
        
        bool is_vertical_lane = (aspect_ratio > 5) && (angle > 30 && angle < 150);
        
        if (is_vertical_lane) {
            drawContours(lane_mask, contours, i, Scalar(255), -1);
        }
    }
    
    binary_image = lane_mask;
}

// 主图像处理函数
tuple<Mat, Mat, Mat, Mat> image_process(const Mat& frame) {
    // 先进行ROI裁剪
    int roi_y = 0;
    int roi_height = static_cast<int>(frame.rows);
    int roi_width = frame.cols;
    
    // 边界保护
    if (roi_y < 0) roi_y = 0;
    if (roi_y >= frame.rows) roi_y = frame.rows - 1;
    if (roi_height < 1) roi_height = 1;
    if (roi_y + roi_height > frame.rows) {
        roi_height = frame.rows - roi_y;
    }
    if (roi_width < 1) roi_width = 1;
    if (roi_width > frame.cols) roi_width = frame.cols;
    
    Rect roi_rect(0, roi_y, roi_width, roi_height);
    Mat frame_cropped = frame(roi_rect).clone();
    
    // 缩放到320x192
    Mat frame_roi;
    resize(frame_cropped, frame_roi, Size(640, 187));

    // 应用gamma校正
    //Mat gamma_corrected_frame;
    //gammaCorrection(frame_roi, gamma_corrected_frame);

    // 高斯模糊
    Mat blurred_frame;
    GaussianBlur(frame_roi, blurred_frame, Size(5, 5), 0);
    
    // 转换到HSV颜色空间
    Mat frame_hsv;
    cvtColor(blurred_frame, frame_hsv, COLOR_BGR2HSV);
    
    // 分离H、S、V通道并应用CLAHE
    vector<Mat> hsv_channels;
    split(frame_hsv, hsv_channels);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(1.5);
    clahe->setTilesGridSize(Size(8, 8));
    Mat v_channel;
    clahe->apply(hsv_channels[2], v_channel);
    hsv_channels[2] = v_channel;
    merge(hsv_channels, frame_hsv);
    
    // 颜色提取
    Mat mask1, mask2, mask3, mask4, red_mask;
    inRange(frame_hsv, Config::lower_red1, Config::upper_red1, mask1);
    inRange(frame_hsv, Config::lower_red2, Config::upper_red2, mask2);
    inRange(frame_hsv, Config::lower_blue, Config::upper_blue, mask3);
    inRange(frame_hsv, Config::lower_yellow, Config::upper_yellow, mask4);
    red_mask = mask1 | mask2;
    
    // 黄色区域处理（去除左右10像素）
    vector<vector<Point>> contours1;
    findContours(mask4, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours1.size(); i++) {
        Rect bounding_rect = boundingRect(contours1[i]);
        if (bounding_rect.width > 20 && bounding_rect.height < 20) {
            rectangle(mask4, Rect(bounding_rect.x, bounding_rect.y, 10, bounding_rect.height), Scalar(0), -1);
            rectangle(mask4, Rect(bounding_rect.x + bounding_rect.width - 10, bounding_rect.y, 10, bounding_rect.height), Scalar(0), -1);
        }
    }
    
    // 二值化处理 - 根据条件选择不同的处理方法
    Mat gray_image;
    cvtColor(blurred_frame, gray_image, COLOR_BGR2GRAY);
    Mat thresholded_image;
    
    // 检查是否需要使用Canny边缘检测
    if (State::sidewalk_end_flag && State::turn_end_flag == 0) {
        // 使用Canny边缘检测
        Mat edges;
        Canny(gray_image, edges, 50, 150);
        
        // 形态学操作强化边缘
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(edges, edges, MORPH_CLOSE, kernel);
        
        // 阈值化处理
        threshold(edges, thresholded_image, 50, 255, THRESH_BINARY);
        
        // 膨胀操作连接断开的边缘
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(thresholded_image, thresholded_image, kernel2);
        
    } else {
        // 使用原来的顶帽变换方法
        // 1. 顶帽变换：提取比背景亮的细节（白线），抑制大面积光照变化（阴影）
        Mat kernel_tophat = getStructuringElement(MORPH_RECT, Size(20, 20));
        Mat top_hat;
        morphologyEx(gray_image, top_hat, MORPH_TOPHAT, kernel_tophat);
        
        // 2. 阈值处理
        threshold(top_hat, thresholded_image, 0, 255, THRESH_BINARY|THRESH_OTSU);
        
        // 3. 形态学操作
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(thresholded_image, thresholded_image, kernel2);
    
        // 应用掩码
        bitwise_not(mask3, mask3);
        bitwise_not(mask4, mask4);
        thresholded_image = thresholded_image & mask3 & mask4;
        bitwise_not(red_mask, red_mask);
        thresholded_image = thresholded_image & red_mask;
    }
    // 轮廓过滤
    vector<vector<Point>> contours;
    findContours(thresholded_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect bounding_rect = boundingRect(contours[i]);
        if (State::sidewalk && bounding_rect.height < 15) {
            rectangle(thresholded_image, bounding_rect, Scalar(0), -1);
        }
        if (!State::sidewalk && (bounding_rect.height < 25 || bounding_rect.width < 20)) {
            rectangle(thresholded_image, bounding_rect, Scalar(0), -1);
        }
    }
    filterLaneLines(thresholded_image);
    return make_tuple(frame_hsv, frame_roi, thresholded_image, thresholded_image);
}

#define LINE_DISTANCE_MIN 50
#define LINE_ANGLE_MIN 200

double TUxiang_Init3(cv::Mat& data)
{    
    int roi_y_start = max(0, (int)(data.rows * 0.3));
    int roi_height = (int)(data.rows * 0.5);
    if (roi_y_start + roi_height > data.rows) {
        roi_height = data.rows - roi_y_start;
    }
    cv::Rect roi_rect(0, roi_y_start, data.cols, roi_height);
    cv::Mat cropped_image = data(roi_rect).clone();
        
    if (cropped_image.cols > 400 || cropped_image.rows > 200) {
        cv::resize(cropped_image, cropped_image, cv::Size(320, 96));
    } else if (cropped_image.cols != 320 || cropped_image.rows != 96) {
        cv::resize(cropped_image, cropped_image, cv::Size(320, 96));
    }
    
    gammaCorrection(cropped_image, cropped_image);
    
    cv::Mat hsv_image;
    cv::cvtColor(cropped_image, hsv_image, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);
    cv::Mat saturation = hsv_channels[1];
    
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(1.5);
    clahe->setTilesGridSize(Size(8, 8));
    
    cv::Mat blur1;
    cv::bilateralFilter(saturation, blur1, 7, 60, 60);

    cv::Mat gaussian_blur1;
    cv::GaussianBlur(blur1, gaussian_blur1, cv::Size(5, 5), 30);
    
    cv::Mat edges;
    cv::Canny(gaussian_blur1, edges, 50, 90);
    
    cv::Mat ca;
    cv::Mat gray_image;
    cv::cvtColor(cropped_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::Mat blur;
    cv::bilateralFilter(gray_image, blur, 7, 60, 60);
    cv::Mat gaussian_blur;
    cv::GaussianBlur(blur, gaussian_blur, cv::Size(5, 5), 30);
    
    cv::Canny(gaussian_blur, ca, 30, 50);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat dilated_ca;
    cv::dilate(ca, dilated_ca, kernel, cv::Point(-1, -1), 2);
    
    std::vector<cv::Vec4i> lines;
    
    cv::HoughLinesP(dilated_ca, lines, 1, CV_PI / 180, 50, 25, 10);

    cv::Mat line_image = cv::Mat::zeros(dilated_ca.size(), CV_8UC1);
   
    cv::Mat line_image2 = cv::Mat::zeros(dilated_ca.size(), CV_8UC1);
    
    cv::Mat dilated_ca2 = cv::Mat::zeros(dilated_ca.size(), CV_8UC1), dilated_caca = cv::Mat::zeros(dilated_ca.size(), CV_8UC1);
    
    std::vector<cv::Vec4i> left_lines, right_lines, left_lines_mid, right_lines_mid;
    
    double left_min_angle = -90;
    double left_max_angle = -18;
    
    double right_min_angle = 18;
    double right_max_angle = 90;
    
    double left_most = 1000, right_most = 0;
    double left_angle = -1000, right_angle = 0;
    
    for (size_t i = 0; i < lines.size(); i ++)
    {
        cv::Vec4i line = lines[i];
        double angle_rad = atan2(line[3] - line[1], line[2] - line[0]);
        double angle_deg = angle_rad * 180 / CV_PI;
        
        if (angle_deg >= left_min_angle && angle_deg <= left_max_angle) {
            left_angle = max(left_angle, angle_deg);
        }
        
        if (angle_deg >= right_min_angle && angle_deg <= right_max_angle) {
            right_angle = max(right_angle, angle_deg);
        }
    }
    
    for (size_t i = 0; i < lines.size(); i ++)
    {
        cv::Vec4i line = lines[i];
        double angle_rad = atan2(line[3] - line[1], line[2] - line[0]);
        double angle_deg = angle_rad * 180 / CV_PI;
        
        if (angle_deg >= left_min_angle && angle_deg <= left_max_angle) {
            if (abs(angle_deg - left_angle) < LINE_ANGLE_MIN) {
                left_lines_mid.push_back(line);
                left_most = cv::min(left_most, (line[0] + line[2]) / 2.0);
            }
        }
    }
    vector<cv::Point> fits_points_l, fits_points_r;
    left_up_line = 0, right_up_line = 0, up_line_count = 0;
    for (size_t i = 0; i < left_lines_mid.size(); i ++)
    {
        cv::Vec4i line = left_lines_mid[i];
        double angle_rad = atan2(line[3] - line[1], line[2] - line[0]);
        double angle_deg = angle_rad * 180 / CV_PI;
    
        if (abs(left_most - (line[0] + line[2]) / 2) < LINE_DISTANCE_MIN)  {
            double slope = tan(angle_rad);
            double b = line[1] - slope * line[0];
            
            vector<cv::Point> pts = extendLine(cv::Point(line[0], line[1]), slope, b, cv::Size(320, 96));
            
            if (pts.size() >= 2) {
                if (pts[0].x + pts[1].x >> 1 >= 270) continue;
                left_lines.push_back(cv::Vec4i(pts[0].x, pts[0].y, pts[1].x, pts[1].y));
            }
            fits_points_l.emplace_back(line[0], line[1]);
            fits_points_l.emplace_back(line[2], line[3]);
        }
    }
    
    for (size_t i = 0; i < left_lines.size(); i++)
    {
        const cv::Vec4i& line = left_lines[i];
        const cv::Point pt1(line[0], line[1]);
        const cv::Point pt2(line[2], line[3]);
        cv::line(line_image, pt1, pt2, cv::Scalar(255), 2, cv::LINE_AA);
    }
    
    if (left_lines.size()) {
        cv::Vec4f line_params;
        cv::fitLine(fits_points_l, line_params, cv::DIST_HUBER, 10, 10, 1);
        
        double vx = line_params[0];
        double vy = line_params[1];
        double x0 = line_params[2];
        double y0 = line_params[3];
        
        double m = vy / vx;
    
        double b2 = y0 - m * x0;
        
        vector<cv::Point> pts2 = extendLine(cv::Point(x0, y0), m, b2, cv::Size(320, 96));
        
        if (pts2.size() >= 2) {
            if (pts2[0].y <= pts2[1].y) {
                left_up_line = pts2[0].x;
            } else {
                left_up_line = pts2[1].x;
            }
            cv::line(line_image2, pts2[0], pts2[1], cv::Scalar(255), 2, cv::LINE_AA);
        } else if (pts2.size() == 1) {
            left_up_line = pts2[0].x;
        } else {
            left_up_line = 0;
        }
    } else {
        left_up_line = 0;
    };
    
    
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i line = lines[i];
        double angle_rad = atan2(line[3] - line[1], line[2] - line[0]);
        double angle_deg = angle_rad * 180 / CV_PI;

        if (angle_deg >= right_min_angle && angle_deg <= right_max_angle) {
            if (abs(angle_deg - right_angle) < LINE_ANGLE_MIN) {
                right_lines_mid.push_back(line);
                right_most = cv::max(right_most, (line[0] + line[2]) / 2.0);
            }
        }
    }
    
    for (size_t i = 0; i < right_lines_mid.size(); i++)
    {
        cv::Vec4i line = right_lines_mid[i];
        double angle_rad = atan2(line[3] - line[1], line[2] - line[0]);
        double angle_deg = angle_rad * 180 / CV_PI;
        
        if (abs(right_most - (line[0] + line[2]) / 2) < LINE_DISTANCE_MIN) {

            double slope = tan(angle_rad);
            double b = line[1] - slope * line[0];
            vector<cv::Point> pts = extendLine(cv::Point(line[0], line[1]), slope, b, cv::Size(320, 96));
            if (pts.size() >= 2) {
                int right_line_center_x = (pts[0].x + pts[1].x) >> 1;
                if (right_line_center_x <= 90) continue;
                
                if (left_up_line > 0 && left_up_line < 320) {
                    int min_right_x = left_up_line + 150;
                    int max_right_x = left_up_line + 250;
                    if (right_line_center_x < min_right_x || right_line_center_x > max_right_x) {
                        continue;
                    }
                } else {
                    if (right_line_center_x > 280) continue;
                }
                
                right_lines.push_back(cv::Vec4i(pts[0].x, pts[0].y, pts[1].x, pts[1].y));
            }
            fits_points_r.emplace_back(line[0], line[1]);
            fits_points_r.emplace_back(line[2], line[3]);
        }
    }
    
    for (size_t i = 0; i < right_lines.size(); i++)
    {
        const cv::Vec4i& line = right_lines[i];
        cv::Point pt1(line[0], line[1]);
        cv::Point pt2(line[2], line[3]);
        
        cv::line(line_image, pt1, pt2, cv::Scalar(255), 2, cv::LINE_AA);
    }

    if (right_lines.size()) {
        cv::Vec4f line_params;
        cv::fitLine(fits_points_r, line_params, cv::DIST_HUBER, 10, 10, 1);
        
        double vx = line_params[0];
        double vy = line_params[1];
        double x0 = line_params[2];
        double y0 = line_params[3];
        
        double m = vy / vx;
    
        double b2 = y0 - m * x0;
        
        vector<cv::Point> pts2 = extendLine(cv::Point(x0, y0), m, b2, cv::Size(320, 96));
        
        if (pts2.size() >= 2) {
            int candidate_right_up_line;
            if (pts2[0].y <= pts2[1].y) {
                candidate_right_up_line = pts2[0].x;
            } else {
                candidate_right_up_line = pts2[1].x;
            }
            
            bool valid_right_line = true;
            if (candidate_right_up_line <= 90) {
                valid_right_line = false;
            } else if (left_up_line > 0 && left_up_line < 320) {
                int lane_width = candidate_right_up_line - left_up_line;
                if (lane_width < 150 || lane_width > 250) {
                    valid_right_line = false;
                }
            } else {
                if (candidate_right_up_line > 280) {
                    valid_right_line = false;
                }
            }
            
            if (valid_right_line) {
                right_up_line = candidate_right_up_line;
                cv::line(line_image2, pts2[0], pts2[1], cv::Scalar(255), 2, cv::LINE_AA);
            } else {
                right_up_line = 0;
            }
        } else if (pts2.size() == 1) {
            int candidate_right_up_line = pts2[0].x;
            bool valid_right_line = true;
            if (candidate_right_up_line <= 90) {
                valid_right_line = false;
            } else if (left_up_line > 0 && left_up_line < 320) {
                int lane_width = candidate_right_up_line - left_up_line;
                if (lane_width < 150 || lane_width > 250) {
                    valid_right_line = false;
                }
            } else {
                if (candidate_right_up_line > 250) {
                    valid_right_line = false;
                }
            }
            
            if (valid_right_line) {
                right_up_line = candidate_right_up_line;
            } else {
                right_up_line = 0;
            }
        } else {
            right_up_line = 0;
        }
    } else {
        right_up_line = 0;
    }
    
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    
    cv::dilate(line_image, dilated_ca2, kernel2, cv::Point(-1, -1), 1);
    cv::dilate(line_image2, dilated_caca, kernel2, cv::Point(-1, -1), 1);
    
#ifdef LINE_DEBUG
    imshow("line_image", dilated_ca2); 
    imshow("line_image2", dilated_caca); 
    imshow("saturation", saturation); 
    imshow("cropped_image", cropped_image);
    imshow("cca", ca);
    waitKey(5);
    return image_handle2(dilated_caca);
#else
    auto point = get_lane_center(left_lines, right_lines);
    double error = point.first - 175;
    double now_error = image_handle2(dilated_caca);
    
    return now_error;
#endif
}

double image_handle2(const Mat data) 
{
    if (left_up_line <= 0 || left_up_line >= data.cols) {
        if (right_up_line > 0 && right_up_line < data.cols && right_up_line != data.cols - 1) {
            left_up_line = right_up_line - 200;
            if (left_up_line < 0) left_up_line = 0;
        } else {
            left_up_line = 0;
        }
    }
    
    if (right_up_line <= 0 || right_up_line >= data.cols || right_up_line == data.cols - 1) {
        if (left_up_line > 0 && left_up_line < data.cols && left_up_line != 0) {
            int estimated_right = left_up_line + 200;
            if (estimated_right >= data.cols) {
                estimated_right = left_up_line + (data.cols - left_up_line) / 2;
                if (estimated_right >= data.cols) estimated_right = data.cols - 1;
            }
            right_up_line = estimated_right;
        } else {
            right_up_line = data.cols - 1;
        }
    }
    
    mid_line = (left_up_line + right_up_line) / 2;
    
    int weight_count = 0;
    
    double error_in = 0, error_out = 0;
    
    int co = data.cols, ro = data.rows;
	std::vector<int> left(ro, -1);
	std::vector<int> right(ro, -1);
	std::vector<int> mid(ro, -1);
	int l_lost = 0;
	int r_lost = 0;
	int t = 0;
    
    int base_value = 160;
    
#ifdef DRAW_MID
    Mat data_cpy = data.clone();
#endif
    
	for (int i = ro - 1 - 10; i >= 10 ; i -= 2) {
		l_lost = 0;
		r_lost = 0;
		for (int j = mid_line; j >= 0; j --) {
			if (data.at<uchar>(i, j) == 255) {
				left[t] = j;
				l_lost = 1;
				break;
			}
		}

		if (l_lost == 0) left[t] = 0;

		for (int j = mid_line; j < co ; j ++) {
			if (data.at<uchar>(i, j) == 255 ) {
				right[t] = j;
				r_lost = 1;
				break;
			}
		}

		if (r_lost == 0) right[t] = co - 1;

#ifdef DRAW_MID
		Point p((left[t] + right[t]) / 2, i);
        circle(data_cpy, p, 2, Scalar(255, 255, 255));
#endif

		mid[t] = (left[t] + right[t]) / 2;
    
        error_in += (mid[t] - base_value) * Weight_th[i * 96 / ro];
    
        weight_count += Weight_th[i * 96 / ro];
		t ++;
    }
    
    error_out = error_in / weight_count;
    
    if (error_out < -160) error_out = -160;
    if (error_out > 160) error_out = 160;
    
    error_out /= 4;

#ifdef DRAW_MID
    imshow("find mid", data_cpy);
#endif

    return error_out;
}

// 旧巡线函数（保留作为备份，但不再使用）
void line_walking_old(const Mat& frame_final) {
    Config::error = 0;

    // 提取巡线 y 范围的图像
    Mat scan_y_wide = frame_final(Range(Config::up_scan_wide, Config::down_scan_wide), Range::all());

    int theory_middle_line = Config::x_middle; // 理论中值
    int left_line_x = Config::x_middle;
    int right_line_x = Config::x_middle;

    Config::draw_middle_line_x[0] = Config::x_middle; // 最后一行从中点开始巡线

    int i = 1;

    // 先找出最后一行的中点
    Mat y_line = scan_y_wide.row(scan_y_wide.rows - 1);
    while (left_line_x > Config::wide_scan + 1) {
        Mat wide_range = y_line(Range::all(), Range(left_line_x - Config::wide_scan, left_line_x));
        if (countNonZero(wide_range) > Config::wide_need) {
            break;
        }
        left_line_x--;
    }

    while (right_line_x < 640 - Config::wide_scan - 1) {
        Mat wide_range = y_line(Range::all(), Range(right_line_x, right_line_x + Config::wide_scan));
        if (countNonZero(wide_range) > Config::wide_need) {
            break;
        }
        right_line_x++;
    }
    Config::draw_left_line_x[0] = left_line_x;
    Config::draw_right_line_x[0] = right_line_x;
    Config::draw_middle_line_x[0] = (right_line_x + left_line_x) / 2;

    // 其余行巡线
    for (int y = scan_y_wide.rows - 1; y >= 0; y -= Config::y_scan_wide) {
        y_line = scan_y_wide.row(y);
        int t = Config::draw_middle_line_x[i - 1];

        while (t > Config::wide_scan + 1) {
            Mat wide_range = y_line(Range::all(), Range(t - Config::wide_scan, t));
            if (countNonZero(wide_range) > Config::wide_need) {
                break;
            }
            left_line_x = t;
            t--;
        }

        t = Config::draw_middle_line_x[i - 1];
        while (t < 640 - Config::wide_scan - 1) {
            Mat wide_range = y_line(Range::all(), Range(t, t + Config::wide_scan));
            if (countNonZero(wide_range) > Config::wide_need) {
                break;
            }
            right_line_x = t;
            t++;
        }

        Config::draw_middle_line_x[i] = (right_line_x + left_line_x) / 2;
        Config::draw_left_line_x[i] = left_line_x;
        Config::draw_right_line_x[i] = right_line_x;
        Config::error += Config::draw_middle_line_x[i] - theory_middle_line;
        i++;
    }

    // 共遍历了 all_count 行中线
    Config::error /= Config::all_count;
}


// PID控制函数
double pid_control(double error) {
    double derivative = error - Config::last_error;
    Config::last_error = error;
    double error_final = (Config::kp * error) + (Config::kd * derivative); // 计算输出值

    // 限制输出值在范围内
    if (error_final >= angle_outmax) {
        return angle_outmax;
    }
    if (error_final < angle_outmin) {
        return angle_outmin;
    }
    return error_final;
}

//变道检测
int turn_detect(cv::Mat frame) {
    if (frame.empty()) {
        return 0; // 0=未检测到
    }

    // 1. 提取ROI区域（与2025.10.7保持一致）
    int roi_x = 120;
    int roi_y = frame.rows *5/ 12;
    int roi_width = frame.cols - 240;
    int roi_height = frame.rows / 2;
    
    if (roi_y < 0 || roi_y + roi_height > frame.rows || roi_width <= 0) {
        return 0; // ROI区域无效
    }
    
    Mat cut_image = frame.clone()(Rect(roi_x, roi_y, roi_width, roi_height));
    
    // 2. 应用gamma校正（2025.10.7的方法）
    gammaCorrection(cut_image, cut_image);
    
    // 3. 转换为HSV颜色空间
    Mat hsv_image;
    cvtColor(cut_image, hsv_image, COLOR_BGR2HSV);
    
    // 4. 创建蓝色掩码（使用高低亮度两套阈值，然后合并）
    // 高亮度蓝色阈值（正常亮度环境）
    Scalar blue_lower_high = Scalar(78, 65, 179);  // H: 95-125, S: 65-255, V: 65-255
    Scalar blue_upper_high = Scalar(133, 255, 255);
    
    // 低亮度蓝色阈值（低亮度环境，降低S和V下限以提高检测率）
    Scalar blue_lower_low = Scalar(103, 158, 88);   // H: 95-125, S: 30-255, V: 30-255（降低S和V下限）
    Scalar blue_upper_low = Scalar(179, 255, 146);
    
    // 创建两个蓝色掩码，然后合并
    Mat Blue1, Blue2;
    inRange(hsv_image, blue_lower_high, blue_upper_high, Blue1);  // 高亮度蓝色
    inRange(hsv_image, blue_lower_low, blue_upper_low, Blue2);    // 低亮度蓝色
    Mat mask_AB_arrow = Blue1 | Blue2;  // 合并两个蓝色掩码
    
    // 5. 形态学操作（2025.10.7的方法）
    Mat kernel = Mat::ones(3, 3, CV_8U);
    erode(mask_AB_arrow, mask_AB_arrow, kernel, Point(-1, -1), 1);
    dilate(mask_AB_arrow, mask_AB_arrow, kernel, Point(-1, -1), 1);
    
    // 6. 查找轮廓
    vector<vector<Point>> contours;
    findContours(mask_AB_arrow, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0; // 未找到轮廓
    }
    
    // 7. 找到面积最大的轮廓（蓝色区域）
    auto max_contour = *max_element(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        });
    
    // 8. 提取ROI区域（2025.10.7的方法：在蓝色区域内查找白色轮廓）
    Rect boundingBox = boundingRect(max_contour);
    Mat roiContour = mask_AB_arrow.clone()(boundingBox);
    
    // 9. 处理ROI区域：将左右边缘的黑色像素变白（2025.10.7的方法）
    for (int row = 0; row < roiContour.rows; row++) {
        for (int col = 0; col < roiContour.cols; col++) {
            if (roiContour.at<uchar>(row, col) == 0) {  // 黑色
                roiContour.at<uchar>(row, col) = 255;  // 变成白色
            } else {
                break;  // 遇到白色，跳出内层循环
            }
        }
    }
    
    for (int row = 0; row < roiContour.rows; row++) {
        for (int col = roiContour.cols - 1; col >= 0; col--) {
            if (roiContour.at<uchar>(row, col) == 0) {  // 黑色
                roiContour.at<uchar>(row, col) = 255;  // 变成白色
            } else {
                break;  // 遇到白色，跳出内层循环
            }
        }
    }
    
    // 10. 反转图像（白色变黑色，黑色变白色，用于查找白色字符）
    roiContour = 255 - roiContour;
    
    // 11. 在ROI内查找白色轮廓
    vector<vector<Point>> contours_roi;
    findContours(roiContour, contours_roi, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours_roi.empty()) {
        return 0; // 未找到白色轮廓
    }
    
    // 12. 找到面积最大的白色轮廓
    auto contour = *max_element(contours_roi.begin(), contours_roi.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        });
    
    double area = contourArea(contour);
    
    // 13. 面积过滤
    if (area <= MIN_AREA_THRESHOLD) {
        return 0; // 面积太小
    }
    
    // 14. 宽高比过滤
    Rect rect = boundingRect(contour);
    float aspect_ratio = (float)rect.width / rect.height;
    if (aspect_ratio <= ASPECT_RATIO_MIN || aspect_ratio >= ASPECT_RATIO_MAX) {
        return 0; // 宽高比不符合要求
    }
    
    // 15. 多边形近似
    vector<Point> approx;
    double epsilon = 0.03 * arcLength(contour, true);
    approxPolyDP(contour, approx, epsilon, true);
    
    if (approx.size() < 4) {
        return 0; // 多边形顶点数不足
    }
    int result =0;
    RotatedRect arrow_rect = minAreaRect(contour);
    float angle = arrow_rect.angle; // 矩形旋转角度
    // 右箭头角度接近 0°/90°，左箭头角度接近 180°/270°
    if (angle > -45 && angle < 45) {
        cout << "左箭头" << endl;
        result = 2;
    } 
    else {
       cout << "右箭头" << endl;
       result = 1;
    }
    return result;
}

/***********************斑马线识别相关******************** */
// 计算点到直线的距离
double pointToLineDistance(const Point2f& point, const Vec4f& line) {
    double x0 = point.x, y0 = point.y;
    double x1 = line[0], y1 = line[1];
    double x2 = line[2], y2 = line[3];
    
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / 
           sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
}

// 检查矩形是否共线
bool checkColinearity(const vector<RotatedRect>& rectangles, Vec4f& bestLine) {
    if (rectangles.size() < 2) return false;
    
    // 提取矩形的中心点
    vector<Point2f> centers;
    for (const auto& rect : rectangles) {
        centers.push_back(rect.center);
    }
    
    // 使用最小二乘法拟合直线
    fitLine(centers, bestLine, DIST_L2, 0, 0.01, 0.01);
    
    // 检查拟合直线的角度是否接近水平（与水平线夹角不超过10度）
    float vx = bestLine[0], vy = bestLine[1];
    double line_angle = atan2(vy, vx) * 180 / CV_PI; // 转换为角度
    
    // 将角度转换到0-180度范围内
    if (line_angle < 0) line_angle += 180;
    if (line_angle > 180) line_angle -= 180;
    
    // 计算与水平线（0度或180度）的最小角度差
    double angle_to_horizontal = min(abs(line_angle - 0), abs(line_angle - 180));
    angle_to_horizontal = min(angle_to_horizontal, abs(line_angle - 90)); // 也考虑90度的情况
    
    // 如果角度偏离水平线超过10度，则返回false
    if (angle_to_horizontal > 20.0) {
        return false;
    }
    
    // 计算所有中心点到拟合直线的距离
    double maxDistance = 0;
    for (const auto& center : centers) {
        double distance = pointToLineDistance(center, bestLine);
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }
    
    return maxDistance <= COLINEAR_THRESHOLD;
}

// 检查矩形的角度是否一致
bool checkAngleConsistency(const vector<RotatedRect>& rectangles) {
    if (rectangles.size() < 2) return true;
    
    double avgAngle = 0;
    for (const auto& rect : rectangles) {
        avgAngle += rect.angle;
    }
    avgAngle /= rectangles.size();
    
    // 检查每个矩形角度与平均角度的差异
    for (const auto& rect : rectangles) {
        double angleDiff = abs(rect.angle - avgAngle);
        // 处理角度环绕问题
        if (angleDiff > 90) angleDiff = 180 - angleDiff;
        if (angleDiff > ANGLE_THRESHOLD) {
            return false;
        }
    }
    
    return true;
}

// 核心函数：输入二值图，识别斑马线（面积最大4个矩形+共线判断）
bool sidewalk_detect(const Mat& frame) {
    if (frame.empty()) {
        return false;
    }

    // 1. ROI区域 - 针对320x180调整
    int roi_x = 120;                     // 从30调整到20
    int roi_y = frame.rows*5 / 12;         // 关注图像下半部分
    int roi_width = frame.cols - 240;    // 从60调整到40
    int roi_height = frame.rows  / 2; // 关注大部分区域
    
    if (roi_y < 0 || roi_y + roi_height > frame.rows || roi_width <= 0) {
        return false;
    }
    
    Mat roi_image = frame(Rect(roi_x, roi_y, roi_width, roi_height));
    Mat processed_image = roi_image.clone();
    
    // 2. 应用gamma校正增强对比度
    gammaCorrection(processed_image, processed_image);
    
    // 3. 转换为灰度图
    Mat gray_image;
    cvtColor(processed_image, gray_image, COLOR_BGR2GRAY);
    
    // 使用Canny边缘检测替代二值化
    Mat edges;
    Canny(gray_image, edges, 50, 150);
    
    // 形态学操作强化边缘
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);
    // 7. 查找轮廓
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return false;
    }
    
    // 8. 过滤轮廓
    vector<RotatedRect> candidate_rectangles;
    vector<double> contour_areas;
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < MIN_AREA_THRESHOLD || area > MAX_AREA_THRESHOLD) continue;
        
        RotatedRect rect = minAreaRect(contour);
        Size2f size = rect.size;
        float aspect_ratio = max(size.width, size.height) / min(size.width, size.height);
        
        if (aspect_ratio >= MIN_ASPECT_RATIO && aspect_ratio <= MAX_ASPECT_RATIO) {
            candidate_rectangles.push_back(rect);
            contour_areas.push_back(area);
        }
    }
    
    // 只需要3个矩形就认为是斑马线
    if (candidate_rectangles.size() < 3) {
        return false;
    }
    
    // 9. 按面积排序，选择面积最大的矩形
    vector<size_t> indices(contour_areas.size());
    for (size_t i = 0; i < contour_areas.size(); ++i) {
        indices[i] = i;
    }
    
    sort(indices.begin(), indices.end(), 
         [&](size_t a, size_t b) { return contour_areas[a] > contour_areas[b]; });
    
    // 选择前3个矩形
    size_t num_rectangles = min(size_t(3), indices.size());
    vector<RotatedRect> selected_rectangles;
    for (size_t i = 0; i < num_rectangles; ++i) {
        selected_rectangles.push_back(candidate_rectangles[indices[i]]);
    }
    
    // 10. 检查矩形是否共线且角度一致
    Vec4f fitted_line;
    bool is_colinear = checkColinearity(selected_rectangles, fitted_line);
    bool angle_consistent = checkAngleConsistency(selected_rectangles);
    
    // 只要有3个矩形，就认为是斑马线
    bool is_zebra_crossing = (selected_rectangles.size() >= 3) && 
                            (is_colinear || angle_consistent);
    
    return is_zebra_crossing;
}

// 蓝色锥桶检测
std::pair<int, int> cone_detect(const Mat& frame_hsv) {
    // 生成蓝色掩码
    Mat blue_mask;
    inRange(frame_hsv, Config::lower_blue, Config::upper_blue, blue_mask);
    blue_mask = blue_mask(Range(Config::blue_mask_down, Config::blue_mask_up), Range::all());

    std::vector<std::vector<Point>> contours;
    findContours(blue_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Config::cone_x = 0;
    Config::cone_y = 0;

    // 查找轮廓并获取蓝色区域中心点坐标
    for (const auto& contour : contours) {
        Rect bounding_rect = boundingRect(contour);
        int x = bounding_rect.x;
        int y = bounding_rect.y;
        int w = bounding_rect.width;
        int h = bounding_rect.height;

        // 检查矩形的宽度和高度是否在设定的范围内
        if (Config::min_width < w && w < Config::max_width && Config::min_height < h && h < Config::max_height) {
            // 计算中心点坐标
            Config::cone_x = x + w / 2;
            Config::cone_y = y + h / 2;
        }
    }

    return {Config::cone_x, Config::cone_y};
}

// 黄色锥桶检测
std::pair<int, int> yellow_cone_detect(const Mat& frame_hsv) {
    // 生成黄色掩码
    Mat yellow_mask;
    inRange(frame_hsv, Config::lower_yellow, Config::upper_yellow, yellow_mask);
    yellow_mask = yellow_mask(Range(Config::blue_mask_down, Config::blue_mask_up), Range::all());

    std::vector<std::vector<Point>> contours;
    findContours(yellow_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Config::cone_x = 0;
    Config::cone_y = 0;

    // 查找轮廓并获取黄色区域中心点坐标
    for (const auto& contour : contours) {
        Rect bounding_rect = boundingRect(contour);
        int x = bounding_rect.x;
        int y = bounding_rect.y;
        int w = bounding_rect.width;
        int h = bounding_rect.height;

        // 检查矩形的宽度和高度是否在设定的范围内
        if (Config::min_width < w && w < Config::max_width && Config::min_height < h && h < Config::max_height) {
            // 计算中心点坐标
            Config::cone_x = x + w / 2;
            Config::cone_y = y + h / 2;
        }
    }

    return {Config::cone_x, Config::cone_y};
}

// 获取车道中心函数（从final_code帮改.cpp移植，简化版）
pair<int, int> get_lane_center(std::vector<cv::Vec4i>& left_lines, std::vector<cv::Vec4i>& right_lines)
{
    double left_x = 0, left_y = 0, right_x = 0, right_y = 0;
    int n = left_lines.size(), m = right_lines.size();
    for (int i = 0; i < n; i ++) {
        int x1 = left_lines[i][0], y1 = left_lines[i][1], x2 = left_lines[i][2], y2 = left_lines[i][3];
        left_x += (x1 + x2) / 2;
        left_y += (y1 + y2) / 2;
    }
    
    left_x /= max(1, n);
    left_y /= max(1, n);
    
    for (int i = 0; i < m; i ++) {
        int x1 = right_lines[i][0], y1 = right_lines[i][1], x2 = right_lines[i][2], y2 = right_lines[i][3];
        right_x += (x1 + x2) / 2;
        right_y += (y1 + y2) / 2;
    }
    
    right_x /= max(1, m);
    right_y /= max(1, m);
    
    if (!m) {
        right_x = 319;
        right_y = 95;
    }
    if (!n){
        left_x = 0;
        left_y = 0;
    }
    int centre_x = (left_x + right_x) / 2;
    int centre_y = (left_y + right_y) / 2;
    
    return pair<int, int>(centre_x, centre_y);
}

//转向处理
void turn_process(){
    Config::turn_detect_res = 1;
    int error_increment = 3.5; // 误差每次增量
    // 左转
    if(Config::turn_detect_res == 1){
        std::cout << "left_turn" << std::endl; 
        int left_and_right_subtract = Config::draw_right_line_x[5] - Config::draw_left_line_x[5]; // 左右边线之差
        
        // 转向误差配置
        Config::error = -15; // 基础误差
        if(Config::turn_time >= 5 && Config::turn_time <= 10) Config::error -= error_increment;
        else if(Config::turn_time > 10 && Config::turn_time <= 15) Config::error -= error_increment;
        else if(Config::turn_time > 15 && Config::turn_time <= 20) Config::error += error_increment;
        else if(Config::turn_time > 20 && Config::turn_time <= 25) Config::error += error_increment;
        else if(Config::turn_time > 25 && Config::turn_time <= 30) Config::error -= error_increment;
        else if(Config::turn_time > 30 && Config::turn_time <= 35) Config::error -= error_increment;
        else if(Config::turn_time > 35 && Config::turn_time <= 40) Config::error += error_increment;
        else if(Config::turn_time > 40 && Config::turn_time <= 45) Config::error += error_increment;
        
        // 转向阶段1：右边线消失，左边线存在
        if(Config::draw_right_line_x[5] == 313 && left_and_right_subtract > 40 && Config::draw_left_line_x[5] >= 150) State::left_exit_flag = 1;
        // 转向阶段2：右边线存在，此时可以退出转向
        if(Config::draw_right_line_x[5] >= 140 && left_and_right_subtract > 40 && State::left_exit_flag == 1 && Config::draw_right_line_x[5] != 313) State::right_exit_flag = 1;
        if(State::right_exit_flag == 1 && State::left_exit_flag == 1){
            State::turn_end_flag = 1;
            Config::turn_time = 0;
        }
    }        
    //右转
    if(Config::turn_detect_res == 2){
        std::cout << "right_turn" << std::endl; 
        int left_and_right_subtract = Config::draw_right_line_x[5] - Config::draw_left_line_x[5]; // 左右边线之差 
        
        // 转向误差配置
        Config::error = 15; // 基础误差
        if(Config::turn_time >= 5 && Config::turn_time <= 10) Config::error += error_increment;
        else if(Config::turn_time > 10 && Config::turn_time <= 15) Config::error += error_increment;
        else if(Config::turn_time > 15 && Config::turn_time <= 20) Config::error -= error_increment;
        else if(Config::turn_time > 20 && Config::turn_time <= 25) Config::error -= error_increment;
        else if(Config::turn_time > 25 && Config::turn_time <= 30) Config::error += error_increment;
        else if(Config::turn_time > 30 && Config::turn_time <= 35) Config::error += error_increment;
        else if(Config::turn_time > 35 && Config::turn_time <= 40) Config::error -= error_increment;
        else if(Config::turn_time > 40 && Config::turn_time <= 45) Config::error -= error_increment;
        
        // 转向阶段1：左边线消失，右边线存在
        if(Config::draw_left_line_x[5] == 7 && left_and_right_subtract > 40 && Config::draw_right_line_x[5] <= 170) State::right_exit_flag = 1;
        // 转向阶段2：左边线存在，此时可以退出转向
        if(Config::draw_left_line_x[5] <= 180 && left_and_right_subtract > 40 && State::right_exit_flag == 1 && Config::draw_left_line_x[5] != 7) State::left_exit_flag = 1;
        if(State::right_exit_flag == 1 && State::left_exit_flag == 1){
            State::turn_end_flag = 1;
            Config::turn_time = 0;
        }

    }
    

}

//蓝色锥桶处理
void cone_process(const Mat& frame_hsv){
    int motor_cone_speed = 201;
    std::tie(Config::cone_x, Config::cone_y) = cone_detect(frame_hsv);
    //检测到锥桶，减速
    if (Config::cone_x != 0) {
        gpioPWM(motor_pin, motor_cone_speed);
        if(State::left_or_right_turn == 0){
            for(int i = 0; i <= 7; i++)
            {
                if(Config::draw_left_line_x[i] == 7) Config::left_lost_time ++;
                if(Config::draw_right_line_x[i] == 313) Config::right_lost_time ++;
            }
            if(Config::left_lost_time > Config::right_lost_time) State::left_or_right_turn = 2; //右绕
            else State::left_or_right_turn = 1; //左绕
        }
    }
    //锥桶第一次出现，count_compare从此刻开始计数
    if (Config::cone_x != 0 && Config::count == 0) {
        Config::count_compare = 0;
    }
    //如果Config.cone_x != 0说明可能出现锥桶，count+1
    if (Config::cone_x != 0) {
        Config::count++;
    }

    //10张图片里如果有8成以上检测到锥桶，则认定为检测到锥桶
    if (Config::count_compare >= 10) {
        if (static_cast<double>(Config::count) / Config::count_compare >= 0.8) {
            State::cone_detect_state_1 = 1;
            if (Config::cone_count == 0) {
                State::cone_start = 1;  // cone_start = 1锥桶开始
            }
        }
        if (static_cast<double>(Config::count) / Config::count_compare <= 0.2 && State::cone_detect_state_1 == 1) {
            State::cone_detect_state_2 = 0;
        }
        //先检测到锥桶，又没检测到，说明经过了第一个锥桶
        if (State::cone_detect_state_1 == 1 && State::cone_detect_state_2 == 0) {
            Config::cone_count++;
            State::cone_detect_state_1 = 0;
            State::cone_detect_state_2 = 1;
        }
        Config::count_compare = 0;
        Config::count = 0;
    }
    if(State::left_or_right_turn == 2){
        // 检测到第一个或第三个锥桶，靠右走
          if (Config::cone_count == 1 && Config::cone_x != 0){
            std::cout << "cone_1" << std::endl;
            Config::error = 0;
            //将锥桶和右下角连接形成直线的斜率k
            double k = (180.0 - (Config::cone_y + Config::blue_mask_down)) / (320 - Config::cone_x);
            //std::cout << std::fixed << std::setprecision(2) << k << std::endl; 
            // y = kx + b 带入(320, 180)后得到的b
            double b = 180 - k * 320;
            for (int i = 0; i < 10; i++) {
                // x = (y - b) / k
                Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
                Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
                Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
            }
            Config::error /= Config::all_count;
        //检测出第二个锥桶，靠左走
          }else if (State::cone_start == 1 && Config::cone_count != 1 && Config::cone_x != 0){
            std::cout << "cone_2" << std::endl;
            Config::error = 0;
            //将锥桶和左下连接形成直线的斜率k
            double k = (180.0 - (Config::cone_y + Config::blue_mask_down)) / (0 - Config::cone_x);
            double b = 180;
            for (int i = 0; i < 10; i++) {
                Config::draw_left_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
                Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
                Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
            }
            Config::error /= Config::all_count;
          }
    }
    if(State::left_or_right_turn == 1){
        // 检测到第一个或第三个锥桶，靠左走
        if (State::cone_start == 1 && Config::cone_count != 1 && Config::cone_x != 0) {
            std::cout << "cone_1" << std::endl;
            Config::error = 0;
            //将锥桶和右下角连接形成直线的斜率k
            double k = (180.0 - (Config::cone_y + Config::blue_mask_down)) / (320 - Config::cone_x);
            //std::cout << std::fixed << std::setprecision(2) << k << std::endl; 
            // y = kx + b 带入(320, 180)后得到的b
            double b = 180 - k * 320;
            for (int i = 0; i < 10; i++) {
                // x = (y - b) / k
                Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
                Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
                Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
            }
            Config::error /= Config::all_count;
        //检测出第二个锥桶，靠右走
        }else if (Config::cone_count == 1 && Config::cone_x != 0) {
            std::cout << "cone_2" << std::endl;
            Config::error = 0;
            //将锥桶和左下连接形成直线的斜率k
            double k = (180.0 - (Config::cone_y + Config::blue_mask_down)) / (0 - Config::cone_x);
            double b = 180;
            for (int i = 0; i < 10; i++) {
                Config::draw_left_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
                Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
                Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
            }
            Config::error /= Config::all_count;
        }
    }
    //State::left_or_right_turn = 0;
}


//AB识别ds版
int simpleDetectAB(const Mat& frame) {
    if (frame.empty()) {
        return 0; // 0=未检测到
    }

    // 1. 提取ROI区域（与2025.10.7保持一致）
    int roi_x = 30;
    int roi_y = frame.rows / 3;
    int roi_width = frame.cols - 60;
    int roi_height = frame.rows / 2;
    
    if (roi_y < 0 || roi_y + roi_height > frame.rows || roi_width <= 0) {
        return 0; // ROI区域无效
    }
    
    Mat cut_image = frame.clone()(Rect(roi_x, roi_y, roi_width, roi_height));
    
    // 2. 应用gamma校正（2025.10.7的方法）
    //gammaCorrection(cut_image, cut_image);
    
    // 3. 转换为HSV颜色空间
    Mat hsv_image;
    cvtColor(cut_image, hsv_image, COLOR_BGR2HSV);
    
    // 4. 创建蓝色掩码（使用高低亮度两套阈值，然后合并）
    // 高亮度蓝色阈值（正常亮度环境）
    Scalar blue_lower_high = Scalar(78, 65, 179);  // H: 95-125, S: 65-255, V: 65-255
    Scalar blue_upper_high = Scalar(133, 255, 255);
    
    // 低亮度蓝色阈值（低亮度环境，降低S和V下限以提高检测率）
    //Scalar blue_lower_low = Scalar(103, 158, 88);   // H: 95-125, S: 30-255, V: 30-255（降低S和V下限）
    //Scalar blue_upper_low = Scalar(179, 255, 146);
    
    // 创建两个蓝色掩码，然后合并
    Mat Blue1, Blue2;
    inRange(hsv_image, blue_lower_high, blue_upper_high, Blue1);  // 高亮度蓝色
    //inRange(hsv_image, blue_lower_low, blue_upper_low, Blue2);    // 低亮度蓝色
    Mat mask_AB_arrow = Blue1 ;  // 合并两个蓝色掩码
    
    // 5. 形态学操作（2025.10.7的方法）
    Mat kernel = Mat::ones(3, 3, CV_8U);
    erode(mask_AB_arrow, mask_AB_arrow, kernel, Point(-1, -1), 1);
    dilate(mask_AB_arrow, mask_AB_arrow, kernel, Point(-1, -1), 1);
    
    // 6. 查找轮廓
    vector<vector<Point>> contours;
    findContours(mask_AB_arrow, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0; // 未找到轮廓
    }
    
    // 7. 找到面积最大的轮廓（蓝色区域）
    auto max_contour = *max_element(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        });
    
    // 8. 提取ROI区域（2025.10.7的方法：在蓝色区域内查找白色轮廓）
    Rect boundingBox = boundingRect(max_contour);
    Mat roiContour = mask_AB_arrow.clone()(boundingBox);
    
    // 9. 处理ROI区域：将左右边缘的黑色像素变白（2025.10.7的方法）
    for (int row = 0; row < roiContour.rows; row++) {
        for (int col = 0; col < roiContour.cols; col++) {
            if (roiContour.at<uchar>(row, col) == 0) {  // 黑色
                roiContour.at<uchar>(row, col) = 255;  // 变成白色
            } else {
                break;  // 遇到白色，跳出内层循环
            }
        }
    }
    
    for (int row = 0; row < roiContour.rows; row++) {
        for (int col = roiContour.cols - 1; col >= 0; col--) {
            if (roiContour.at<uchar>(row, col) == 0) {  // 黑色
                roiContour.at<uchar>(row, col) = 255;  // 变成白色
            } else {
                break;  // 遇到白色，跳出内层循环
            }
        }
    }
    
    // 10. 反转图像（白色变黑色，黑色变白色，用于查找白色字符）
    roiContour = 255 - roiContour;
    
    // 11. 在ROI内查找白色轮廓
    vector<vector<Point>> contours_roi;
    findContours(roiContour, contours_roi, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours_roi.empty()) {
        return 0; // 未找到白色轮廓
    }
    
    // 12. 找到面积最大的白色轮廓
    auto contour = *max_element(contours_roi.begin(), contours_roi.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        });
    
    double area = contourArea(contour);
    
    // 13. 面积过滤
    if (area <= MIN_AREA_THRESHOLD) {
        return 0; // 面积太小
    }
    
    // 14. 宽高比过滤
    Rect rect = boundingRect(contour);
    float aspect_ratio = (float)rect.width / rect.height;
    if (aspect_ratio <= ASPECT_RATIO_MIN || aspect_ratio >= ASPECT_RATIO_MAX) {
        return 0; // 宽高比不符合要求
    }
    
    // 15. 多边形近似
    vector<Point> approx;
    double epsilon = 0.03 * arcLength(contour, true);
    approxPolyDP(contour, approx, epsilon, true);
    
    if (approx.size() < 4) {
        return 0; // 多边形顶点数不足
    }
    
    // 16. 凸包检测AB（2025.10.7的核心方法）
    vector<Point> hull;
    convexHull(approx, hull);
    double hull_area = contourArea(hull);
    double contour_area = contourArea(approx);
    
    // 17. 计算面积比例并判断
    double area_ratio = contour_area / hull_area;
    int result = 0;
    
    if (area_ratio > AB_AREA_RATIO_THRESHOLD) {
        result = 2; // B
        std::cout<<"b"<<std::endl;
    } else {
        result = 1; // A
        std::cout<<"a"<<std::endl;
    }
    return result;
}

// AB停车处理
void AB_stop(int result){
	   Config::error = 0;

     if(result==2){ //右停
         for (int i = 0; i < 10; i++) {
             //Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
             Config::draw_left_line_x[i] += 80;
             Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
             Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
         }
     }else if(result==1){ //左停
         for (int i = 0; i < 10; i++) {
             //Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
             Config::draw_left_line_x[i] -= 80;
             Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
             //Config::draw_middle_line_x[i] -= 40;
             Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
         }
     }
     Config::error /= Config::all_count;
     
}
/************************************
有关锥桶转向
************************************/
// 检测蓝色锥桶（无绘图）
bool detectBlueCones(const Mat& frame, Mat& blue_mask) {
    Mat hsv, roi_image;
    
    // 提取ROI区域
    Rect roi_rect(0, 200, 640, 320);
    if (roi_rect.x + roi_rect.width > frame.cols) roi_rect.width = frame.cols - roi_rect.x;
    if (roi_rect.y + roi_rect.height > frame.rows) roi_rect.height = frame.rows - roi_rect.y;
    
    roi_image = frame(roi_rect).clone();
    
    // 颜色检测
    cvtColor(roi_image, hsv, COLOR_BGR2HSV);
    inRange(hsv, Config::lower_blue, Config::upper_blue, blue_mask);
    
    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(blue_mask, blue_mask, kernel);
    dilate(blue_mask, blue_mask, kernel);
    
    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(blue_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int blue_cone_count = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        Rect rect = boundingRect(contours[i]);
        
        // 计算宽高比
        float aspect_ratio = 1.0f;
        if (rect.height > 0) {
            aspect_ratio = static_cast<float>(rect.height) / static_cast<float>(rect.width);
        }
        
        // 过滤条件
        if (area > MIN_CONTOUR_AREA && area < MAX_CONTOUR_AREA &&
            aspect_ratio >= MIN_CONE_ASPECT_RATIO && aspect_ratio <= MAX_CONE_ASPECT_RATIO) {
            blue_cone_count++;
        }
    }
    
    return (blue_cone_count >= MIN_BLUE_CONES_COUNT);
}
//转向角度计算
int angleToPwm(int angle) {
    // 限制角度在有效范围内，防止舵机过载
    if (angle < SERVO_MIN) angle = SERVO_MIN;
    if (angle > SERVO_MAX) angle = SERVO_MAX;
    
    // 公式推导：0.5ms对应6，2.5ms对应32，线性映射角度到PWM
    // 角度每增加1°，PWM增加 (32-6)/180 ≈ 0.144
    return 58 + (int)(angle * 0.633);  // 通用公式，可根据舵机手册微调
}



// 提取黄色锥桶中心点（无绘图）
vector<Point2f> getConeCenters(const Mat& frame, Mat& yellow_mask) {
    Mat hsv, roi_image;
    
    // 提取ROI区域
    Rect roi_rect(0, 200, 640, 320);
    if (roi_rect.x + roi_rect.width > frame.cols) roi_rect.width = frame.cols - roi_rect.x;
    if (roi_rect.y + roi_rect.height > frame.rows) roi_rect.height = frame.rows - roi_rect.y;
    
    roi_image = frame(roi_rect).clone();
    
    // 颜色检测
    cvtColor(roi_image, hsv, COLOR_BGR2HSV);
    inRange(hsv, Config::lower_yellow, Config::upper_yellow, yellow_mask);
    
    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(yellow_mask, yellow_mask, kernel);
    dilate(yellow_mask, yellow_mask, kernel);
    
    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(yellow_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<Point2f> centers;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        Rect rect = boundingRect(contours[i]);
        
        // 计算宽高比（避免除零）
        float aspect_ratio = 1.0f;
        if (rect.height > 0) {
            aspect_ratio = static_cast<float>(rect.height) / static_cast<float>(rect.width);
        }
        
        // 过滤条件：面积+宽高比
        if (area > MIN_CONTOUR_AREA && area < MAX_CONTOUR_AREA &&
            aspect_ratio >= MIN_CONE_ASPECT_RATIO && aspect_ratio <= MAX_CONE_ASPECT_RATIO) {
            
            // 计算中心点
            Moments m = moments(contours[i]);
            if (m.m00 != 0) {
                Point2f center(m.m10/m.m00, m.m01/m.m00);
                centers.push_back(center);
            }
        }
    }
    
    return centers;
}

// 最小二乘法拟合直线（y = kx + b）
bool fitLineToCones(vector<Point2f> centers, float &k, float &b) {
    if (centers.size() < 2) return false;

    double sumX=0, sumY=0, sumXY=0, sumX2=0;
    int n = centers.size();
    for (auto &p : centers) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;
    }

    double denom = n*sumX2 - sumX*sumX;
    if (fabs(denom) < 1e-6) return false;

    k = (n*sumXY - sumX*sumY) / denom;
    b = (sumY - k*sumX) / n;
    return true;
}

// 根据引导线斜率k计算目标转向角度
int calculateTargetAngle(double k) {
    if (fabs(k) >3.7||fabs(k)<DEAD_ZONE){
        k = 0;
    }
    // 1. 死区处理：k绝对值过小时，保持直行（避免频繁微调）
    if (fabs(k) < DEAD_ZONE) {
        return 90;
    }
    // 2. 核心映射逻辑：
    // - k为正：引导线向右偏（锥桶右偏）→ 需要向右转（角度增大，靠近SERVO_MAX）
    // - k为负：引导线向左偏（锥桶左偏）→ 需要向左转（角度减小，靠近SERVO_MIN）
    int targetAngle;
    if(k<0){
        targetAngle = - (int)(k * K_TO_ANGLE);
    }
    else if(k>=0){
        targetAngle = 180 - (int)(k * K_TO_ANGLE);
    }
    // 3. 限制角度在安全范围（防止超出舵机物理极限）
    if (targetAngle < SERVO_MIN) targetAngle = SERVO_MIN;
    if (targetAngle > SERVO_MAX) targetAngle = SERVO_MAX;
    
    return targetAngle;
}
// 锥桶检测主函数（无绘图）
bool detectYellowCones(const Mat& frame, vector<Point2f>& centers, float& angle, bool& has_blue,bool& had_blue) {
    if (frame.empty()) return false;
    
    Mat gamma_image;
    gammaCorrection(frame, gamma_image);
    
    Mat blue_mask, yellow_mask;
    
    // 第一步：检测蓝色锥桶
    has_blue = detectBlueCones(gamma_image, blue_mask);
    
    // 如果检测到蓝色锥桶，直接返回未检测到黄色锥桶
    if (has_blue) {
        std::cout<<"有蓝色"<<std::endl;
        centers.clear();
        angle = 90;
        had_blue = true;
        return false;
    }
    
    // 第二步：检测黄色锥桶（未检测到蓝色锥桶时）
    centers = getConeCenters(gamma_image, yellow_mask);
    
    // 只有检测到≥3个锥桶才进行后续处理
    bool detected = (centers.size() >= MIN_CONES_COUNT);
    float k=0, b=0;
    
    if (detected && fitLineToCones(centers, k, b)) {
        // 计算转向角度
        angle = calculateTargetAngle(k);
    } else {
        angle = 90;
    }
    
    return detected;
}
/**************************************************************************************** */

int main(){

    int program_run_count = 0;
    
    LaneChangeState lane_state = {0, {}, {}, {}};

    int AB_stop_time = 0;
    // 电机速度参数（展示图像时车停止）
    int motor_speed = 210;// 191 -206，原210
    int motor_ab_speed = 205;//用于ab检测和左右检测的速度
    int motor_cone_speed = 200;  // 过锥桶速度适当减速
    int motor_yellow_cone_speed = 208;  // 过锥桶速度适当减速
    int motor_ABstop_speed = 201;

    // ===================== 新增：黄色锥桶减速相关变量 =====================
    // 黄色锥桶处理完成后的时间戳
    std::chrono::time_point<std::chrono::high_resolution_clock> yellow_cone_finish_time;
    // 标记是否已触发减速（避免重复执行）
    bool yellow_cone_decelerate_triggered = false;
    // 延迟减速时间（秒）
    const int yellow_cone_decelerate_delay = 4;
    // =====================================================================

    // 添加帧率控制变量
    auto last_time = std::chrono::high_resolution_clock::now();
    const double target_fps = 30.0;  // 目标帧率30FPS

    // 新增：时间记录变量（用于30秒保存逻辑）
    std::chrono::time_point<std::chrono::high_resolution_clock> last_save_time;
    bool save_time_initialized = false;  // 标记是否初始化过保存时间

    // 注册Ctrl+C信号处理器（关键）
    std::cout << "注册Ctrl+C信号处理器..." << std::endl;
    signal(SIGINT, signalHandler);
    std::cout << "现在可以使用 Ctrl+C 安全退出程序" << std::endl;

    // 创建输入监听线程
    pthread_t threadId;
    pthread_create(&threadId, nullptr, listener, nullptr);

    // 初始化电机和舵机
    
    if (gpioInitialise() < 0) {
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 设置电机
    gpioSetMode(motor_pin, PI_OUTPUT);
    gpioSetPWMfrequency(motor_pin, 500);
    gpioPWM(motor_pin, 191); // 倒转最大128，正转最大255
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 设置舵机
    gpioSetMode(servo_pin, PI_OUTPUT);
    gpioSetPWMfrequency(servo_pin, 300);
    gpioPWM(servo_pin, servo_middle);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 初始化双缓冲摄像头（添加详细日志和错误处理）
    std::cout << "正在打开双缓冲摄像头设备0..." << std::endl;
    DoubleBufferCamera* cap = nullptr;
    try {
        cap = new DoubleBufferCamera(CAMERA_PORT);
        std::cout << "双缓冲摄像头已成功打开，分辨率: " << CAMERA_WIDTH << "x" << CAMERA_HEIGHT << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误：无法打开摄像头 - " << e.what() << std::endl;
        // 尝试释放可能被占用的资源
        system("sudo fuser -k /dev/video0 2>/dev/null");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        try {
            cap = new DoubleBufferCamera(CAMERA_PORT);
            std::cout << "双缓冲摄像头重试成功打开" << std::endl;
        } catch (const std::exception& e2) {
            std::cerr << "错误：重试后仍无法打开摄像头，退出程序" << std::endl;
            gpioTerminate();
            return 1;
        }
    }
    
    // 注册全局摄像头指针（关键：用于信号处理器中断）
    g_cap_ptr.store(cap);
    
    waiting(*cap);
    
    // 如果在等待阶段就收到退出指令，直接清理
    if (!running) {
        std::cout << "等待阶段收到退出指令，开始清理..." << std::endl;
        gpioPWM(motor_pin, 128);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // 等待线程退出，设置超时
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 2; // 2秒超时
        pthread_timedjoin_np(threadId, nullptr, &timeout);
        
        // 释放双缓冲摄像头（带强制清理）
        if (cap != nullptr) {
            cap->release();
            delete cap;
            cap = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        system("sudo fuser -k /dev/video0 2>/dev/null");
        
        gpioTerminate();
        system("sudo killall -9 pigpiod 2>/dev/null");
        return 0;
    }
    
    while (running) {
        // 在每次循环开始时检查退出标志（关键）
        if (!running) {
            std::cout << "[1] 循环开始检测到退出标志" << std::endl;
            break;
        }

        cv::Mat frame;
        
        // 关键修复：读取前再次检查running标志
        if (!running) {
            std::cout << "[2] 摄像头读取前检测到退出标志" << std::endl;
            break;
        }
        
        // 使用双缓冲摄像头读取（非阻塞，性能更优）
        if (!cap->read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (frame.empty()) {
            // 关键修复：如果是因为Ctrl+C导致的空帧，直接退出
            if (!running) {
                std::cout << "[空帧检测] 检测到退出标志，停止处理" << std::endl;
                break;
            }
            
            std::cout << "警告：获取空帧..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // ===================== 新增：30秒保存逻辑开始 =====================
        // 初始化保存时间和目录（仅执行一次）
        if (!save_time_initialized) {
            last_save_time = std::chrono::high_resolution_clock::now();
            save_time_initialized = true;

            // 检查保存目录是否存在，不存在则创建（Linux/macOS）
            struct stat info;
            if (stat(IMAGE_SAVE_DIR, &info) != 0) {
                if (mkdir(IMAGE_SAVE_DIR, 0755) != 0) {
                    std::cerr << "警告：无法创建保存目录 " << IMAGE_SAVE_DIR << std::endl;
                } else {
                    std::cout << "创建保存目录成功：" << IMAGE_SAVE_DIR << std::endl;
                }
            }
        }

        // 计算当前时间与上次保存时间的差值
        auto save_current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(save_current_time - last_save_time).count();

        // 达到30秒间隔，执行保存
        if (elapsed_seconds >= IMAGE_SAVE_INTERVAL_SEC) {
            // 克隆原始帧（避免修改处理中的帧）
            cv::Mat save_frame = frame.clone();

            // 处理时间戳：转换为东八区（+8小时）
            auto now_sys = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now_sys);
            time_t_now += 8 * 3600;  // 东八区时差
            std::tm* timeinfo = std::gmtime(&time_t_now);

            // 格式化时间戳字符串（用于绘制和文件名）
            std::ostringstream time_str, filename_str;
            time_str << std::setfill('0') << (timeinfo->tm_year + 1900) << "-"
                     << std::setw(2) << (timeinfo->tm_mon + 1) << "-"
                     << std::setw(2) << timeinfo->tm_mday << " "
                     << std::setw(2) << timeinfo->tm_hour << ":"
                     << std::setw(2) << timeinfo->tm_min << ":"
                     << std::setw(2) << timeinfo->tm_sec;

            // 在图像左上角绘制绿色时间戳
            cv::putText(save_frame, time_str.str(), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            // 生成带时间戳的文件名
            filename_str << IMAGE_SAVE_DIR << "frame_"
                         << (timeinfo->tm_year + 1900) << std::setw(2) << (timeinfo->tm_mon + 1)
                         << std::setw(2) << timeinfo->tm_mday << "_"
                         << std::setw(2) << timeinfo->tm_hour << std::setw(2) << timeinfo->tm_min
                         << std::setw(2) << timeinfo->tm_sec << ".jpg";

            // 保存图像
            if (cv::imwrite(filename_str.str(), save_frame)) {
                std::cout << "[保存图像] " << filename_str.str() << std::endl;
            } else {
                std::cerr << "警告：保存图像失败 " << filename_str.str() << std::endl;
            }

            // 更新上次保存时间，开始下一个周期
            last_save_time = save_current_time;
        }
        // 检查退出标志（移到循环条件中）
        if (!running) {
            std::cout << "[3] 图像处理前检测到退出标志，开始停车..." << std::endl;
            break;
        }

        // 图像处理（保留用于其他功能如斑马线检测、锥桶检测等）
        cv::Mat frame_hsv, frame_320x180, frame_final, frame_red;
        std::tie(frame_hsv, frame_320x180, frame_final, frame_red) = image_process(frame);
        

        cv::Mat frame_for_lane = frame.clone();
        double lane_error = TUxiang_Init3(frame_for_lane);
        Config::error = static_cast<int>(lane_error * 3);
        // 斑马线检测+-
        // 修改检测条件，使其在适当的时候检测斑马线
        // 添加自动检测机制，不仅依赖cone_count，也根据距离和时间判断
        static int frame_counter = 0;
        frame_counter++;
        
        // 在特定条件下自动检测斑马线
        // 添加调试信息：每30帧输出一次检测状态
        if (frame_counter % 30 == 0) {
            std::cout << "[斑马线检测] cone_count=" << Config::cone_count 
                      << " | sidewalk_end_flag=" << State::sidewalk_end_flag 
                      << " | frame=" << frame_counter << std::endl;
        }
        
        // 斑马线检测：每帧都检测，确保不错过
        if ((Config::cone_count >= 1 && Config::cone_count <= 2) && !State::sidewalk_end_flag) {
           bool temp = sidewalk_detect(frame);
           
           // 添加调试输出
           if (frame_counter % 30 == 0) {
               std::cout << "[斑马线检测] 检测结果: " << (temp ? "检测到" : "未检测到") << std::endl;
           }
           
           if(temp == true){
                std::cout << "===== 检测到斑马线，立即停车 =====" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(110));
                gpioPWM(motor_pin, 128); // 立即停止电机
                std::this_thread::sleep_for(std::chrono::seconds(1));
                system("aplay -D hw:1,0 /usr/share/sounds/alsa/tyq.wav"); // 播放提示音
                std::this_thread::sleep_for(std::chrono::seconds(4));
                State::sidewalk = true;
                State::sidewalk_end_flag = true;
           }
        }

        // 转向处理
        // 注释掉斑马线条件：不再需要先检测到斑马线才能右转
        if (State::sidewalk_end_flag && State::turn_end_flag == 0) {
        //if (State::turn_end_flag == 0) {  // 只需要转向未结束即可
            //turn_process();
            // 转向时减速，提高稳定性
            //Config::turn_detect_res = turn_detect(frame_320x180);
            gpioPWM(motor_pin, motor_ab_speed);  // 假设motor_low_speed是低速值
            Config::turn_detect_res = 2;
            if(Config::turn_detect_res!=0){
                gpioPWM(motor_pin, motor_speed);  // 假设motor_low_speed是低速值
                //std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 减速缓冲
                // 统一执行右转：使用140°角度
                int target_angle = 135;  // 右转角度
                // 由于turn_detect现在统一返回2（右转），所以统一使用右转角度
                if(Config::turn_detect_res==1){
                    target_angle = 135;  // 左箭头也统一右转
                }
                else if(Config::turn_detect_res==2){
                    target_angle = 45;  // 右转角度
                }
                gpioPWM(servo_pin, angleToPwm(target_angle));
                std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                std::this_thread::sleep_for(std::chrono::milliseconds(600));
                
                gpioPWM(servo_pin, angleToPwm(180-target_angle));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                
                // 恢复电机速度，标记转向结束
                gpioPWM(motor_pin, motor_speed);
                Config::turn_time += 1; 
                State::turn_end_flag = 1;
            }
            
        }
        // 蓝色锥桶处理
        if (Config::cone_count < 1) {
           cone_process(frame_hsv);
        }
        //黄色锥桶第一次处理
        if(State::yellow_cone_end == 0 && State::turn_end_flag == 1){
            State::yellow_cone_start=1;
            std::cout<<"返回弯道检测开始"<<std::endl;
            gpioPWM(motor_pin, motor_yellow_cone_speed); 
            vector<Point2f> centers;
            float zhuitong_angle = 90;
            bool zhuitong_has_blue = false;
            bool zhuitong_detected = detectYellowCones(frame, centers, zhuitong_angle, zhuitong_has_blue, zhuitong_had_blue);
            if(Config::turn_detect_res==1){
                    zhuitong_angle = 45;
                }
            else if(Config::turn_detect_res==2){
                zhuitong_angle = 135; 
            }
            //if (zhuitong_detected && !zhuitong_has_blue) {
            if (zhuitong_had_blue && !zhuitong_has_blue) { 
                std::cout<<"没有蓝色"<<std::endl;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
                gpioPWM(servo_pin, angleToPwm(zhuitong_angle));
                //转向延时
                std::this_thread::sleep_for(std::chrono::milliseconds(500));// 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                std::this_thread::sleep_for(std::chrono::milliseconds(600));
                gpioPWM(servo_pin, angleToPwm(180-zhuitong_angle));
                std::this_thread::sleep_for(std::chrono::milliseconds(300));  // 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                
                // ===================== 关键修改1：记录黄色锥桶处理完成时间 =====================
                State::yellow_cone_end=1;
                yellow_cone_finish_time = std::chrono::high_resolution_clock::now(); // 记录完成时间
                yellow_cone_decelerate_triggered = false; // 重置减速标志
                std::cout << "[黄色锥桶] 处理完成，开始计时" << yellow_cone_decelerate_delay << "秒后减速到motor_ab_speed" << std::endl;
                // ============================================================================
            }
        }

        // ===================== 关键修改2：黄色锥桶4秒后减速逻辑 =====================
        // 黄色锥桶处理完成后，未触发减速，且已过指定延迟时间
        if (State::yellow_cone_end == 1 && !yellow_cone_decelerate_triggered) {
            auto yellow_current_time = std::chrono::high_resolution_clock::now();
            // 计算已过去的秒数
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                yellow_current_time - yellow_cone_finish_time
            ).count();
            
            // 达到4秒延迟，执行减速
            if (elapsed_seconds >= yellow_cone_decelerate_delay) {
                std::cout << "[黄色锥桶延时减速] 已过" << elapsed_seconds << "秒，电机速度从当前值调整为：" << motor_ab_speed << std::endl;
                gpioPWM(motor_pin, motor_ab_speed); // 减速到motor_ab_speed
                yellow_cone_decelerate_triggered = true; // 标记已减速，避免重复执行
            } else if (elapsed_seconds % 1 == 0) { // 每秒输出一次倒计时（可选）
                std::cout << "[黄色锥桶延时减速] 倒计时：" << (yellow_cone_decelerate_delay - elapsed_seconds) << "秒后减速" << std::endl;
            }
        }
        // ============================================================================

        // AB区域停车
        if (State::yellow_cone_end==1 && State::AB_flag==0){
            std::cout<<"ab检测开始"<<std::endl;
            AB_frame_count++;

            int result2 = simpleDetectAB(frame);   // 0=未检测到，1=A，2=B

            // 只记录有效结果（1或2），最多存3个
            if ((result2 == 1 || result2 == 2) && AB_detect_count < 3) {
                AB_detect_results[AB_detect_count] = result2;
                AB_detect_count++;
            }

            bool should_decide = false;

            // 情况1：已经采样满3次，按多数投票
            if (AB_detect_count == 3) {
                int countA = 0, countB = 0;
                for (int i = 0; i < AB_detect_count; i++) {
                    if (AB_detect_results[i] == 1) countA++;
                    else if (AB_detect_results[i] == 2) countB++;
                }
                // 多数票；平票时默认偏向第二个结果
                if (countA > countB) final_result = 1;
                else if (countB > countA) final_result = 2;
                else final_result = AB_detect_results[1]; // 平局时按第二次

                should_decide = true;
            }
            // 情况2：在限定帧数内始终采样不到3次，但有至少1次有效结果
            else if (AB_frame_count >= AB_MAX_FRAMES && AB_detect_count > 0) {
                // 不足3次时，优先取“第二次”的结果；如果只有1次，就用第一次
                int index = (AB_detect_count >= 2) ? 1 : 0;
                final_result = AB_detect_results[index];
                should_decide = true;
            }

            if (should_decide) {
                State::AB_flag = 1;

                // 详细输出AB识别结果
                std::cout << "\n========================================" << std::endl;
                if(final_result == 1){
                    std::cout << "===== AB投票结果：A区停车位，执行左停车 =====" << std::endl;
                } else if(final_result == 2){
                    std::cout << "===== AB投票结果：B区停车位，执行右停车 =====" << std::endl;
                }
                std::cout << "已采样次数: " << AB_detect_count << " ，帧数: " << AB_frame_count << std::endl;
                std::cout << "AB区域坐标: (" << Config::AB_x << ", " << Config::AB_y << ")" << std::endl;
                std::cout << "电机速度切换为: " << motor_ABstop_speed << std::endl;
                std::cout << "AB_flag已置位，开始停车流程" << std::endl;
                std::cout << "========================================\n" << std::endl;
            }
        }
        if(State::AB_flag == 1&&final_result!=0){
            AB_stop(final_result);
        }

        double error_final = pid_control(Config::error / 3);
        std::cout << error_final << std::endl;
        //std::cout << "sidewalk:" << State::sidewalk << std::endl;
            if(program_run_count == 10) gpioPWM(servo_pin, servo_middle - error_final);
            
            // ===================== 关键修改3：兼容减速逻辑，避免覆盖速度 =====================
            // 仅在未触发黄色锥桶减速时，才使用原motor_speed
            if(Config::cone_x == 0 && program_run_count == 10 && State::AB_flag == 0 && !yellow_cone_decelerate_triggered) {
                gpioPWM(motor_pin, motor_speed);
            }
            // ================================================================================
            
            if(State::AB_flag == 1) gpioPWM(motor_pin, motor_ABstop_speed);
            Config::count_compare++;
            program_run_count ++;
        
            if(program_run_count >= 10)  program_run_count = 10;

        // 帧率控制 - 控制处理速度以适应高速场景
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = current_time - last_time;
        long long elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double target_duration = 1000000.0 / target_fps;  // 微秒
        
        if (elapsed_microseconds < target_duration) {
            double sleep_time = target_duration - elapsed_microseconds;
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(sleep_time)));
        }
        
        last_time = std::chrono::high_resolution_clock::now();

        //std::cout << Config::count_compare << std::endl;
        
        // 检查退出按键（ESC或q）
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {  // 按下 ESC 或 q 退出
            std::cout << "检测到退出按键，设置退出标志..." << std::endl;
            running = false;
            break;
        }
    }
    
    // 统一的清理流程
    std::cout << "\n========== 开始清理退出流程 ==========" << std::endl;
    
    // 1. 停止电机
    std::cout << "1. 停止电机..." << std::endl;
    gpioPWM(motor_pin, 128);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // 2. 舵机回中
    std::cout << "2. 舵机回中..." << std::endl;
    gpioPWM(servo_pin, servo_middle);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    // 3. 释放双缓冲摄像头（关键修复：确保完全释放）
    std::cout << "3. 释放双缓冲摄像头..." << std::endl;
    if (cap != nullptr) {
        cap->release();
        delete cap;
        cap = nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 等待内核驱动释放设备
    }
    
    // 强制释放Linux设备文件句柄（关键）
    std::cout << "   强制释放/dev/video0设备锁..." << std::endl;
    system("sudo fuser -k /dev/video0 2>/dev/null");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    // 清空全局指针
    g_cap_ptr.store(nullptr);
    
    // 4. 等待监听线程退出（带超时）
    std::cout << "4. 等待监听线程退出..." << std::endl;
    running = false; // 确保线程能退出
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 3; // 3秒超时
    
    int join_result = pthread_timedjoin_np(threadId, nullptr, &timeout);
    if (join_result == ETIMEDOUT) {
        std::cout << "警告：监听线程超时未退出，强制取消..." << std::endl;
        pthread_cancel(threadId);
        pthread_join(threadId, nullptr);
    } else if (join_result == 0) {
        std::cout << "监听线程已正常退出" << std::endl;
    }
    
    // 5. 终止GPIO
    std::cout << "5. 终止GPIO控制..." << std::endl;
    gpioTerminate();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // 6. 清理pigpiod进程
    std::cout << "6. 清理pigpiod进程..." << std::endl;
    system("sudo killall -9 pigpiod 2>/dev/null");
    
    std::cout << "========== 退出流程完成 ==========\n" << std::endl;
    
    return 0;
}