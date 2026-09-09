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
    static const int x_middle = 160; // 理论中线
    static const int wide_scan = 5; // x方向每次取一行的5个像素点
    static const int wide_need = 2; // 5个像素点≥2，则检测到边线
    static const int up_scan_wide = 130; // 从图像的130行开始巡线
    static const int down_scan_wide = 170; // 从图像的170行结束巡线
    static const int y_scan_wide = 5; // y方向5行检测一次边线
    static std::array<uint16_t, 10> draw_middle_line_x; // 存储检测到的中线点
    static std::array<uint16_t, 10> draw_left_line_x; // 存储检测到的左边线点
    static std::array<uint16_t, 10> draw_right_line_x; // 存储检测到的右边线点
    static int error; // 误差
    static int last_error; // 记录上一次误差
    static const double kp;
    static const double kd;
    static const int all_count; // 共遍历all_count行

    // 斑马线检测参数
    static const int sidewalk_detect_y = 160; // 斑马线固定行检测
    static const int sidewalk_len = 50; // 左右边线差小于50则检测到斑马线
    // 斑马线检测核心参数（类内初始化，仅一次）
    static constexpr int white_pixel_strong_thresh = 190; // 强特征阈值
    static constexpr int white_pixel_normal_thresh = 140; // 普通特征阈值
    static constexpr int line_gap_thresh = 80; // 跑道宽度阈值
    static constexpr int sidewalk_min_continuous = 2; // 连续行数
    static constexpr int sidewalk_scan_range[2] = {100, 180}; // 数组用constexpr
    static constexpr int sidewalk_scan_step = 1; // 遍历步长

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
int Config::turn_detect_res = 0;

int Config::AB_x = 0;
int Config::AB_y = 0;

int turn_line = 5;

bool State::sidewalk = false;
bool State::sidewalk_end_flag = false;//原false
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

int State::yellow_cone_start=0;//黄色锥桶第一次返回开始标志
int State::yellow_cone_end=0;//黄色锥桶第一次返回结束标志
int State::yellow_cone_start_2=0;//黄色锥桶第二次检测开始标志
int State::yellow_cone_end_2=0;//黄色锥桶第二次检测结束标志

// // 2. 初始化参数（确保生效）
// const int Config::white_pixel_strong_thresh = 190;
// const int Config::white_pixel_normal_thresh = 140;
// const int Config::line_gap_thresh = 80;
// const int Config::sidewalk_min_continuous = 2;
// const int Config::sidewalk_scan_range[2] = {100, 180};
// const int Config::sidewalk_scan_step = 2;

float Config::yellow_cone_k=0;
float Config::yellow_cone_b=0;


// 电机和舵机gpio引脚
const int motor_pin = 13;
const int servo_pin = 12;

// 舵机转向参数
const int servo_middle = 115; //115,原120
const int servo_min = 77;  // 最右边
const int servo_max = 153;  // 最左边
int angle_outmax = 8;  // 9
int angle_outmin = -8; // -9
const int K_TO_ANGLE = 20;      // k到角度的映射系数（初始值，需校准）
const int DEAD_ZONE = 0.1;         // k的死区（小于此值不转向，避免抖动）
const int SERVO_MIN = 40;    //角度最小值（对应右转）
const int SERVO_MAX = 140;   //角度最大值（对应左转）

// 视频帧长宽
const int frame_width = 320;
const int frame_height = 180;
int asd = 0;
// 用于标记主程序的运行状态
std::atomic<bool> running(true);

// 输入监听线程的函数
void* listener(void* arg) {
    char buffer[256];
    while (running) {
        // 检查终端输入，这里使用 fgets
        if (fgets(buffer, sizeof(buffer), stdin)) {
            // 去掉换行符
            buffer[strcspn(buffer, "\n")] = 0;
            std::cout << "Received input: " << buffer << std::endl;
            
            if (strcmp(buffer, "s") == 0) {
                State::sidewalk = true;
            }
            else if (strcmp(buffer, "b") == 0) {
                State::turn_end_flag = true;
            }
            else {
                running = false;
            }
            //if (asd) running = false;
            //if (strcmp(buffer, "a") == 0) {
            //    Config::AB_x = 1;
            //    asd = 1;
            //}
        }
    }
    return nullptr;
}

// 发车等待
void waiting(cv::VideoCapture& cap) {
    while(true){
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cap.release();
            cap.open(-1);
            cap >> frame;
            continue;
        }
        // 调整图像大小
        Mat resized_frame;
        resize(frame, resized_frame, Size(320, 180));
        // 转换到 HSV 颜色空间
        Mat frame_hsv;
        cvtColor(resized_frame, frame_hsv, COLOR_BGR2HSV);
        // 生成蓝色掩码
        Mat blue_mask;
        inRange(frame_hsv, Config::lower_blue, Config::upper_blue, blue_mask);
        // 蓝色小于50%启动
        if (cv::countNonZero(blue_mask) < (320 * 90) ) {
            cout << "start" << cv::countNonZero(blue_mask) << endl;
            break;
        }
        cout << "waiting" << endl;
    }
}


// 图像处理
std::tuple<Mat, Mat, Mat, Mat> image_process(const Mat& frame) {
    // 调整图像大小
    Mat resized_frame;
    resize(frame, resized_frame, Size(320, 180));

    // 高斯模糊
    Mat blurred_frame;
    GaussianBlur(resized_frame, blurred_frame, Size(5, 5), 0);

    // 转换到 HSV 颜色空间
    Mat frame_hsv;
    cvtColor(blurred_frame, frame_hsv, COLOR_BGR2HSV);

    // 进行颜色提取
    Mat frame_final;
    Mat mask1, mask2, mask3, mask4;
    inRange(frame_hsv, Config::lower_red1, Config::upper_red1, mask1);
    inRange(frame_hsv, Config::lower_red2, Config::upper_red2, mask2);
    inRange(frame_hsv, Config::lower_blue, Config::upper_blue, mask3);
    inRange(frame_hsv, Config::lower_yellow, Config::upper_yellow, mask4);
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(mask3, mask3, kernel1, Point(-1, -1), 2);
    dilate(mask4, mask4, kernel1, Point(-1, -1), 1);
    
    std::vector<std::vector<cv::Point>> contours1;
    cv::findContours(mask4, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours1.size(); i++) {
        // 计算轮廓的边界框
        cv::Rect bounding_rect = cv::boundingRect(contours1[i]);
        if (bounding_rect.width > 20 && bounding_rect.height < 20){
            // 涂黑轮廓左边10个像素
            cv::rectangle(mask4, cv::Rect(bounding_rect.x, bounding_rect.y, 10, bounding_rect.height), cv::Scalar(0, 0, 0), -1);
    
            // 涂黑轮廓右边10个像素
            cv::rectangle(mask4, cv::Rect(bounding_rect.x + bounding_rect.width - 10, bounding_rect.y, 10, bounding_rect.height), cv::Scalar(0, 0, 0), -1);
        }
    }
    
    frame_final = mask1 | mask2 | mask3;
    
    //bitwise_not(frame_red, frame_red);
    cv::Mat image_final;
    bitwise_and(blurred_frame, blurred_frame, image_final);
    cv::Mat gray_image;
    cv::cvtColor(image_final, gray_image, cv::COLOR_BGR2GRAY);

    // 应用自适应阈值
    cv::Mat thresholded_image;
    cv::adaptiveThreshold(gray_image, thresholded_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 0);

    // 定义结构元素
    cv::Mat kernel2 = cv::Mat::ones(4, 4, CV_8U);

    // 应用开运算：先进行腐蚀再膨胀
    cv::Mat dilated_image;
    cv::dilate(thresholded_image, dilated_image, kernel2, cv::Point(-1, -1), 1);

    // 对阈值图像进行位取反
    cv::bitwise_not(mask3, mask3);
    cv::bitwise_not(mask4, mask4);
    thresholded_image = thresholded_image & mask3 &mask4;
    //cv::bitwise_not(thresholded_image, thresholded_image);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 遍历所有轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓的面积
        double area = cv::contourArea(contours[i]);
        cv::Rect bounding_rect = cv::boundingRect(contours[i]);
        // 如果轮廓的面积小于30个像素，将其涂黑
        //if (area < 30) {
        //    cv::drawContours(thresholded_image, contours, i, cv::Scalar(0, 0, 0), -1);
        //}
        if (State::sidewalk && bounding_rect.height < 15) {
            // 涂黑轮廓
            cv::rectangle(thresholded_image, bounding_rect, cv::Scalar(0, 0, 0), -1);
        }
        if (!State::sidewalk && bounding_rect.height < 25 || bounding_rect.width < 20) {
            cv::rectangle(thresholded_image, bounding_rect, cv::Scalar(0, 0, 0), -1);
        }
    }


    // 闭运算：先膨胀后腐蚀
    //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    //morphologyEx(thresholded_image, thresholded_image, MORPH_CLOSE, kernel);
    //dilate(thresholded_image, thresholded_image, kernel, Point(-1, -1), 1);
    //morphologyEx(frame_red, frame_red, MORPH_CLOSE, kernel);
    //dilate(frame_red, frame_red, kernel, Point(-1, -1), 1);
    

    return std::make_tuple(frame_hsv, resized_frame, thresholded_image, thresholded_image);
}

// 巡线函数
void line_walking(const Mat& frame_final) {
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

    while (right_line_x < 320 - Config::wide_scan - 1) {
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
        while (t < 320 - Config::wide_scan - 1) {
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
int turn_detect(Mat image, bool debug) {
    // 读取图像(640, 480)
    Rect roi(160, 350, 320, 130); // 获取感兴趣区域
    image = image(roi);

    // 转换为 HSV 色彩空间
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);

    // 创建蓝色掩膜
    Mat blue_mask;
    inRange(hsv_image, Config::lower_blue, Config::upper_blue, blue_mask);
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 5));
    // dilate(blue_mask, blue_mask, kernel, Point(-1, -1), 1);

    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(blue_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 筛选出最大的轮廓作为识别区域
    if (contours.empty()) {
        return 0; // 未找到蓝色区域
    }

    // 过滤出点数超过16的轮廓
    vector<vector<Point>> filtered_contours;
    for (const auto& contour : contours) {
        if (contour.size() > 16) {
            filtered_contours.push_back(contour);
        }
    }
    if (filtered_contours.size() >= 2){
        // 排序，以便找到最大的两个轮廓
        sort(filtered_contours.begin(), filtered_contours.end(), 
            [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) > contourArea(b); // 按面积降序排序
            });
    }
    vector<Point> merged_contour;
    // 检查是否存在至少两个符合条件的轮廓
    if (filtered_contours.size() >= 2) {
        // 获取最大的两个轮廓
        vector<Point> largest_contour = filtered_contours[0]; // 最大轮廓
        vector<Point> second_largest_contour = filtered_contours[1]; // 第二大轮廓

        // 合并两个轮廓
        merged_contour.insert(merged_contour.end(), largest_contour.begin(), largest_contour.end());
        merged_contour.insert(merged_contour.end(), second