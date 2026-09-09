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
    static constexpr int white_pixel_strong_thresh = 200; // 强特征阈值
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
        merged_contour.insert(merged_contour.end(), second_largest_contour.begin(), second_largest_contour.end());
        cout << "two_contours" << endl;
    }
    else if (filtered_contours.size() == 1) {
        merged_contour = filtered_contours[0];
        cout << "one_contour" << endl;
    }
    
    // 将轮廓点转换为二维数组
    vector<Point> points = merged_contour;

    // 使用线性规划确定四个点
    Point left_top = *max_element(points.begin(), points.end(),
        [](const Point& a, const Point& b) { return -a.x - 2 * a.y < -b.x - 2 * b.y; });
    Point left_bottom = *max_element(points.begin(), points.end(),
        [](const Point& a, const Point& b) { return -a.x + a.y < -b.x + b.y; });
    Point right_top = *max_element(points.begin(), points.end(),
        [](const Point& a, const Point& b) { return a.x - a.y < b.x - b.y; });
    Point right_bottom = *max_element(points.begin(), points.end(),
        [](const Point& a, const Point& b) { return a.x + 2 * a.y < b.x + 2 * b.y; });

    // 将角点分别存储在一个数组中
    Point2f src_points[4] = {left_top, left_bottom, right_top, right_bottom};

    // 定义目标矩形的四个角点
    int width = 210, height = 297;
    Point2f dst_points[4] = {
        Point2f(0, 0),
        Point2f(0, height - 1),
        Point2f(width - 1, 0),
        Point2f(width - 1, height - 1)
    };

    // 进行透视变换
    Mat matrix = getPerspectiveTransform(src_points, dst_points);
    Mat warped;
    warpPerspective(blue_mask, warped, matrix, Size(width, height));

    // 找中线
    int w_start = 0, w_end = 0;
    for (int i = 0; i < 209; i++) {
        if (warped.at<uchar>(180, i) && !warped.at<uchar>(180, i + 1)) {
            w_start = i;
            break;
        }
    }
    for (int i = w_start + 1; i < 209; i++) {
        if (warped.at<uchar>(180, i)) {
            w_end = i;
            break;
        }
    }
    int mid = (w_start + w_end) / 2;

    if (debug) {
        cout << left_top << endl;
        cout << left_bottom << endl;
        cout << right_bottom << endl;
        cout << right_top << endl;
        // circle(image, left_top, 2, Scalar(0, 255, 0), -1);
        // circle(image, left_bottom, 2, Scalar(0, 255, 0), -1);
        // circle(image, right_top, 2, Scalar(0, 255, 0), -1);
        // circle(image, right_bottom, 2, Scalar(0, 255, 0), -1);
        for (int i = 0; i < 4; ++i) {
        // 绘制角点
        circle(image, src_points[i], 2, Scalar(0, 0, 255), -1);
        }
        imwrite("blue_mask.jpg", blue_mask);
        imwrite("img.jpg", image);
        imwrite("Warped_Image.jpg", warped);
    }

    // 分割图像区域
    Mat left_half = warped(Rect(0, 0, mid, height));
    bitwise_not(left_half, left_half);
    Mat right_half = warped(Rect(mid, 0, width - mid, height));
    bitwise_not(right_half, right_half);

    // 计算左右半边的白色区域面积
    int left_area = countNonZero(left_half);
    int right_area = countNonZero(right_half);

    // 输出结果
    if (left_area > right_area) {
        if (debug) {
            cout << "左半边白色区域更大，面积为 " << left_area << " 像素大于右边的 " << right_area << " 像素。" << endl;
        }
        return 1;
    } else {
        if (debug) {
            cout << "右半边白色区域更大，面积为 " << right_area << " 像素大于左边的 " << left_area << " 像素。" << endl;
        }
        return 2;
    }
}

// 斑马线检测
int sidewalk_detect(const Mat& frame_final, cv::VideoCapture& cap) {
     static std::deque<int> local_white; 
    bool is_zebra = false;
    int full_rows = frame_final.rows;

    // 1. 高速场景：聚焦斑马线最后出现的105-120行（从文档数据提炼的核心区域）
    int scan_start = 105;
    int scan_end = 120;
    scan_start = std::max(0, scan_start);
    scan_end = std::min(full_rows - 1, scan_end);
    std::cout << "高速检测区域：行" << scan_start << "到行" << scan_end << std::endl;

    local_white.clear();
    // 2. 新增：单次触发防抖计数（避免正常跑道的单帧噪声干扰）
    static int trigger_buffer = 0; 

    for (int y = scan_start; y <= scan_end; y += 1) {
        Mat y_line = frame_final.row(y);
        int white_count = countNonZero(y_line);
        int left_line_x = Config::x_middle;
        int right_line_x = Config::x_middle;

        // 3. 复用原边线计算逻辑（文档数据中偏移行不影响强特征行判断）
        while (left_line_x > Config::wide_scan + 1) {
            Mat wide_range = y_line(Range::all(), Range(left_line_x - Config::wide_scan, left_line_x));
            if (countNonZero(wide_range) > Config::wide_need) break;
            left_line_x--;
        }
        while (right_line_x < 320 - Config::wide_scan - 1) {
            Mat wide_range = y_line(Range::all(), Range(right_line_x, right_line_x + Config::wide_scan));
            if (countNonZero(wide_range) > Config::wide_need) break;
            right_line_x++;
        }
        int line_gap = right_line_x - left_line_x;
        std::cout << "行" << y << "：白色=" << white_count << "，宽度=" << line_gap << std::endl;

        // 4. 强特征判断（基于文档数据的斑马线特征：白色≥170+宽度45-70）
        bool zebra_strong = (white_count >= 170) && (line_gap >= 45 && line_gap <= 70);

        // 5. 【插入位置】防抖逻辑：基于强特征更新计数，1行即触发（适配高速短暂特征）
        if (zebra_strong) {
            trigger_buffer++;
            // 文档数据中斑马线强特征仅1-2行，故计数≥1即触发，兼顾速度与防抖
            if (trigger_buffer >= 1) { 
                is_zebra = true;
                trigger_buffer = 0; // 重置计数，避免重复触发
                std::cout << "触发：行" << y << "（白色=" << white_count << "，宽度=" << line_gap << "）" << std::endl;
                break;
            }
        } else {
            // 非强特征行，计数递减（过滤单帧噪声）
            trigger_buffer = std::max(0, trigger_buffer - 1);
        }
    }

    // 6. 高速停车逻辑（文档数据中需快速响应，故缩短缓冲）
    if (is_zebra) {
        std::cout << "===== 高速模式：检测到斑马线，立即停车 =====" << std::endl;
        gpioPWM(motor_pin, 128);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        system("aplay -D hw:1,0 /usr/share/sounds/alsa/abc.wav");
        std::this_thread::sleep_for(std::chrono::seconds(4));
        State::sidewalk_end_flag = true;
        local_white.clear();
    }

    return Config::turn_detect_res;
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
    State::left_or_right_turn = 0;
}

/*********************************************************************************************************************************************
 * 检测特定颜色的锥桶
 * @param hsv 输入的HSV图像（便于颜色分割）
 * @param lower 颜色的HSV下界
 * @param upper 颜色的HSV上界
 * @param cones 输出参数：存储检测到的锥桶中心坐标
 */
 /*
void detectColorCones(const Mat& hsv, const Scalar& lower, const Scalar& upper, vector<Point2f>& cones) {
    cones.clear();
    Mat mask;
    
    inRange(hsv, lower, upper, mask);
    
    // 形态学操作去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    double min_contour_area = 50.0;
    
    for (const auto& contour : contours) {
        if (contourArea(contour) > min_contour_area) {
            Moments m = moments(contour);
            if (m.m00 != 0) {  // 修正：使用m.m00而不是m00
                double cx = m.m10 / m.m00;  // 修正：使用m.m00
                double cy = m.m01 / m.m00;  // 修正：使用m.m00
                cones.push_back(Point2f(cx, cy));
            }
        }
    }
}

void detectAllCones(const Mat& frame, vector<Point2f>& blue_cones, vector<Point2f>& yellow_cones, bool show_debug) {
    blue_cones.clear();
    yellow_cones.clear();
    
    Mat hsv;  // 移除参数遮蔽，使用局部变量
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    // 检测蓝色锥桶：将array转换为Scalar
    detectColorCones(hsv, 
                    cv::Scalar(Config::lower_blue[0], Config::lower_blue[1], Config::lower_blue[2]), 
                    cv::Scalar(Config::upper_blue[0], Config::upper_blue[1], Config::upper_blue[2]), 
                    blue_cones);
    // 检测黄色锥桶：将array转换为Scalar
    detectColorCones(hsv, 
                    cv::Scalar(Config::lower_yellow[0], Config::lower_yellow[1], Config::lower_yellow[2]), 
                    cv::Scalar(Config::upper_yellow[0], Config::upper_yellow[1], Config::upper_yellow[2]), 
                    yellow_cones);
    
    if (show_debug) {
        Mat debug_frame = frame.clone();
        // 绘制蓝色锥桶
        for (const auto& cone : blue_cones) {
            circle(debug_frame, cone, 6, Scalar(255, 0, 0), -1); // 蓝色
            circle(debug_frame, cone, 10, Scalar(255, 0, 0), 2);
        }
        // 绘制黄色锥桶
        for (const auto& cone : yellow_cones) {
            circle(debug_frame, cone, 6, Scalar(0, 255, 255), -1); // 黄色
            circle(debug_frame, cone, 10, Scalar(0, 255, 255), 2);
        }
        imshow("Cone Detection", debug_frame);
    }
}

int recognizeLaneDirection(const vector<Point2f>& blue_cones, const vector<Point2f>& yellow_cones, int img_width, int img_height) {
    if (blue_cones.empty() || yellow_cones.empty()) {
        return 0; // unknown
    }
    
    // 计算蓝色和黄色锥桶的平均位置
    Point2f blue_center(0, 0), yellow_center(0, 0);
    
    for (const auto& cone : blue_cones) {
        blue_center += cone;
    }
    blue_center.x /= blue_cones.size();
    blue_center.y /= blue_cones.size();
    
    for (const auto& cone : yellow_cones) {
        yellow_center += cone;
    }
    yellow_center.x /= yellow_cones.size();
    yellow_center.y /= yellow_cones.size();
    
    // 转换为相对坐标 (0-1范围)
    Point2f blue_rel(blue_center.x / img_width, blue_center.y / img_height);
    Point2f yellow_rel(yellow_center.x / img_width, yellow_center.y / img_height);
    
    // 根据规则判断变道方向
    // 左变道: 蓝色在内侧下方，黄色在外侧上方
    // 右变道: 蓝色在内侧上方，黄色在外侧下方
    
    if (blue_rel.y > yellow_rel.y) {
        // 蓝色在黄色下方 -> 左变道
        return 1;
    } else {
        // 蓝色在黄色上方 -> 右变道
        return 2;
    }
}

bool validateConePattern(int direction, const vector<Point2f>& blue_cones, const vector<Point2f>& yellow_cones, int img_width, int img_height) {
    if (blue_cones.size() < 2 || yellow_cones.size() < 2) {
        return false;
    }
    
    // 计算锥桶分布特征
    vector<double> blue_x, blue_y, yellow_x, yellow_y;
    for (const auto& cone : blue_cones) {
        blue_x.push_back(cone.x / img_width);
        blue_y.push_back(cone.y / img_height);
    }
    for (const auto& cone : yellow_cones) {
        yellow_x.push_back(cone.x / img_width);
        yellow_y.push_back(cone.y / img_height);
    }
    
    // 计算平均值 - 使用std::accumulate
    double blue_avg_x = std::accumulate(blue_x.begin(), blue_x.end(), 0.0) / blue_x.size();
    double blue_avg_y = std::accumulate(blue_y.begin(), blue_y.end(), 0.0) / blue_y.size();
    double yellow_avg_x = std::accumulate(yellow_x.begin(), yellow_x.end(), 0.0) / yellow_x.size();
    double yellow_avg_y = std::accumulate(yellow_y.begin(), yellow_y.end(), 0.0) / yellow_y.size();
    
    if (direction == 1) { // 左变道
        // 左变道验证: 蓝色在0.25-0.6x, 黄色在0.5-0.8x
        return (blue_avg_x >= 0.25 && blue_avg_x <= 0.6 && 
               yellow_avg_x >= 0.5 && yellow_avg_x <= 0.8 &&
               blue_avg_y > yellow_avg_y); // 蓝色在黄色下方
    } else if (direction == 2) { // 右变道
        // 右变道验证: 蓝色在0.25-0.6x, 黄色在0.5-0.8x
        return (blue_avg_x >= 0.25 && blue_avg_x <= 0.6 && 
               yellow_avg_x >= 0.5 && yellow_avg_x <= 0.8 &&
               blue_avg_y < yellow_avg_y); // 蓝色在黄色上方
    }
    
    return false;
}

vector<Point2f> sortConesByY(const vector<Point2f>& cones, bool ascending) {
    vector<Point2f> sorted = cones;
    sort(sorted.begin(), sorted.end(), 
         [ascending](const Point2f& a, const Point2f& b) {
             return ascending ? (a.y < b.y) : (a.y > b.y);
         });
    return sorted;
}

vector<Point2f> generateGuidePath(int direction, const vector<Point2f>& blue_cones, const vector<Point2f>& yellow_cones) {
    vector<Point2f> path;
    
    if (blue_cones.empty() || yellow_cones.empty()) {
        return path;
    }
    
    // 根据变道方向采用不同的路径生成策略
    if (direction == 1) { // 左变道
        // 左变道路径：从蓝色锥桶到黄色锥桶的平滑过渡
        vector<Point2f> sorted_blue = sortConesByY(blue_cones, false); // 降序
        vector<Point2f> sorted_yellow = sortConesByY(yellow_cones, false); // 降序
        
        // 生成路径点：在蓝色和黄色锥桶之间取中点
        for (size_t i = 0; i < min(sorted_blue.size(), sorted_yellow.size()); i++) {
            Point2f midpoint((sorted_blue[i].x + sorted_yellow[i].x) / 2, 
                           (sorted_blue[i].y + sorted_yellow[i].y) / 2);
            path.push_back(midpoint);
        }
    } else if (direction == 2) { // 右变道
        // 右变道路径：从蓝色锥桶到黄色锥桶的平滑过渡
        vector<Point2f> sorted_blue = sortConesByY(blue_cones, true); // 升序
        vector<Point2f> sorted_yellow = sortConesByY(yellow_cones, true); // 升序
        
        // 生成路径点
        for (size_t i = 0; i < min(sorted_blue.size(), sorted_yellow.size()); i++) {
            Point2f midpoint((sorted_blue[i].x + sorted_yellow[i].x) / 2, 
                           (sorted_blue[i].y + sorted_yellow[i].y) / 2);
            path.push_back(midpoint);
        }
    }
    
    return path;
}

double calculatePathError(const vector<Point2f>& path_points, int img_width, int img_height) {
    if (path_points.empty()) {
        return 0.0;
    }
    
    // 找到距离车辆最近的点（图像底部）
    Point2f vehicle_pos(img_width / 2, img_height);
    
    // 查找最近点
    Point2f nearest_point = path_points[0];
    double min_distance = norm(vehicle_pos - nearest_point);
    
    for (const auto& point : path_points) {
        double distance = norm(vehicle_pos - point);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_point = point;
        }
    }
    
    // 计算横向误差
    double error = nearest_point.x - vehicle_pos.x;
    
    return error;
}
void drawDebugInfo(Mat& frame, const vector<Point2f>& blue_cones, const vector<Point2f>& yellow_cones, 
                  const vector<Point2f>& guide_path, double error, int direction, int img_width, int img_height) {
    Mat display_frame = frame.clone();
    
    // 绘制锥桶
    for (const auto& cone : blue_cones) {
        circle(display_frame, cone, 4, Scalar(255, 0, 0), -1); // 蓝色
    }
    for (const auto& cone : yellow_cones) {
        circle(display_frame, cone, 4, Scalar(0, 255, 255), -1); // 黄色
    }
    
    // 绘制引导路径
    for (size_t i = 0; i < guide_path.size(); i++) {
        circle(display_frame, guide_path[i], 3, Scalar(0, 255, 0), -1); // 绿色路径点
        if (i > 0) {
            line(display_frame, guide_path[i-1], guide_path[i], Scalar(0, 255, 0), 2);
        }
    }
    
    // 绘制车辆位置和误差
    Point vehicle_center(img_width / 2, img_height);
    Point error_point(vehicle_center.x + error, vehicle_center.y - 50);
    
    circle(display_frame, vehicle_center, 8, Scalar(0, 0, 255), -1); // 红色车辆
    arrowedLine(display_frame, vehicle_center, error_point, Scalar(255, 0, 255), 3); // 误差向量
    
    // 显示信息
    string dir_text = "Direction: ";
    switch (direction) {
        case 1: dir_text += "LEFT"; break;
        case 2: dir_text += "RIGHT"; break;
        default: dir_text += "UNKNOWN"; break;
    }
    
    putText(display_frame, dir_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    string error_text = "Error: " + to_string(int(error));
    putText(display_frame, error_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    imshow("Cone Following", display_frame);
}
//***************************************************************************************************************************************** */
//AB识别ds版
std::string simpleDetectAB(const Mat& frame_hsv) {
    if (frame_hsv.empty()) return "Unknown";
    
    // 快速预处理
    cv::Mat gray, binary;
    if (frame_hsv.channels() == 3) {
        cv::cvtColor(frame_hsv, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame_hsv;
    }
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY_INV);
    
    // 轮廓分析（最快的方法）
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return "Unknown";
    
    auto largest_contour = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
    
    double area = cv::contourArea(*largest_contour);
    std::vector<cv::Point> hull;
    cv::convexHull(*largest_contour, hull);
    double hull_area = cv::contourArea(hull);
    
    if (hull_area > 0) {
        double solidity = area / hull_area;
        return (solidity < 0.75) ? "A" : "B";
    }
    
    return "Unknown";
}
//AB识别db版
int letter_detect(const Mat& frame_hsv, bool debug) {
    if (frame_hsv.empty()) {
        if (debug) cout << "输入图像为空！" << endl;
        return 0;
    }
    
    // 动态计算ROI：取图像下方1/3高度，中间1/2宽度
    int img_width = frame_hsv.cols;
    int img_height = frame_hsv.rows;
    
    int roi_x = img_width / 4; // 左边界：图像1/4宽度处
    int roi_y = img_height * 2 / 3; // 上边界：图像2/3高度处（下方区域）
    int roi_width = img_width / 2; // 宽度：图像1/2
    int roi_height = img_height / 3; // 高度：图像1/3
    
    // 再次检查ROI是否合法（双重保险）
    if (roi_x < 0 || roi_y < 0 || 
        roi_x + roi_width > img_width || 
        roi_y + roi_height > img_height) {
        if (debug) cout << "ROI超出图像范围，自动调整为图像大小！" << endl;
        // 若仍不合法，直接取整幅图像作为ROI
        Rect roi(0, 0, img_width, img_height);
    }
    
    Rect roi(roi_x, roi_y, roi_width, roi_height);
    Mat roi_image = frame_hsv(roi).clone();
    // -------------------------- 1. 复用原框架：ROI提取（排除背景干扰） --------------------------
    // 注意：需根据字母在赛道的实际位置调整ROI（x,y,宽,高），原箭头ROI可能不适用
    //Rect roi(100, 300, 300, 180); // 示例ROI：需实际测试调整
    //Mat roi_image = frame_hsv(roi).clone(); // 克隆避免修改原图像


    // -------------------------- 2. 复用原框架：颜色掩膜（分割红色字母） --------------------------
    Mat hsv_image;
    cvtColor(roi_image, hsv_image, COLOR_BGR2HSV); // BGR转HSV，抗光照干扰

    // 红色跨HSV的0度，需合并两个掩膜
    Mat red_mask1, red_mask2, red_mask;
    inRange(hsv_image, Config::lower_red1, Config::upper_red1, red_mask1);   // 低饱和度红色
    inRange(hsv_image, Config::lower_red2, Config::upper_red2, red_mask2); // 高饱和度红色
    bitwise_or(red_mask1, red_mask2, red_mask); // 合并两个红色掩膜

    // （可选）形态学操作：消除小噪声，让字母轮廓更完整
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(red_mask, red_mask, kernel);  // 腐蚀：去小噪声
    dilate(red_mask, red_mask, kernel); // 膨胀：恢复字母轮廓


    // -------------------------- 3. 复用原框架：轮廓筛选（保留字母主体） --------------------------
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(red_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 筛选条件：1. 轮廓点数>20（排除小噪声）；2. 轮廓面积>500（确保是字母，非杂点）
    vector<vector<Point>> valid_contours;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (contour.size() > 20 && area > 500) {
            valid_contours.push_back(contour);
        }
    }

    // 未找到有效轮廓：返回未识别
    if (valid_contours.empty()) {
        if (debug) cout << "未检测到字母轮廓" << endl;
        return 0;
    }

    // 取面积最大的轮廓（假设画面中只有一个字母）
    int max_idx = 0;
    double max_area = contourArea(valid_contours[0]);
    for (int i = 1; i < valid_contours.size(); i++) {
        double area = contourArea(valid_contours[i]);
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }
    vector<Point> letter_contour = valid_contours[max_idx];


    // -------------------------- 4. 复用原框架：透视变换（矫正字母视角） --------------------------
    // 步骤4.1：获取字母轮廓的“最小包围矩形”（用于确定透视变换的4个源点）
    RotatedRect min_rect = minAreaRect(letter_contour);
    Point2f src_points[4]; // 透视变换的源点（包围矩形的4个角）
    min_rect.points(src_points);

    // 步骤4.2：定义透视变换的目标矩形（正视角，与模板尺寸一致）
    int dst_width = 210, dst_height = 297; // 与标准模板尺寸统一
    Point2f dst_points[4] = {
        Point2f(0, 0),                  // 左上
        Point2f(0, dst_height - 1),     // 左下
        Point2f(dst_width - 1, dst_height - 1), // 右下
        Point2f(dst_width - 1, 0)       // 右上
    };

    // 步骤4.3：执行透视变换，得到正视角的字母图像
    Mat perspective_matrix = getPerspectiveTransform(src_points, dst_points);
    Mat warped_letter; // 矫正后的字母图像（二值图）
    warpPerspective(red_mask, warped_letter, perspective_matrix, Size(dst_width, dst_height));

    // 转为纯二值图（0=黑背景，255=白字母）
    threshold(warped_letter, warped_letter, 127, 255, THRESH_BINARY);

    cout << "待匹配图像尺寸：宽=" << warped_letter.cols << ", 高=" << warped_letter.rows << endl;
    // -------------------------- 5. 核心修改：形状匹配（区分A和B） --------------------------
    // 步骤5.1：加载提前准备的A、B标准模板
    Mat template_A = imread("/home/pi/5G_car/template_A.jpg", IMREAD_GRAYSCALE);
    Mat template_B = imread("/home/pi/5G_car/template_B.jpg", IMREAD_GRAYSCALE);
    if (template_A.empty() || template_B.empty()) {
        if (debug) cout << "模板加载失败！请检查template_A.png和template_B.png路径" << endl;
        return 0;
    }
    cout << "A模板尺寸：宽=" << template_A.cols << ", 高=" << template_A.rows << endl;
    cout << "B模板尺寸：宽=" << template_B.cols << ", 高=" << template_B.rows << endl;
    // 步骤5.2：模板匹配（计算待识别字母与A、B模板的相似度）
    Mat result_A, result_B;
    // 使用归一化相关系数匹配（TM_CCOEFF_NORMED），结果越接近1，相似度越高
    matchTemplate(warped_letter, template_A, result_A, TM_CCOEFF_NORMED);
    matchTemplate(warped_letter, template_B, result_B, TM_CCOEFF_NORMED);

    // 提取匹配结果的最大值（相似度）
    double max_val_A, max_val_B;
    minMaxLoc(result_A, nullptr, &max_val_A, nullptr, nullptr);
    minMaxLoc(result_B, nullptr, &max_val_B, nullptr, nullptr);

    // 步骤5.3：调试信息输出
    if (debug) {
        cout << "与A模板的相似度：" << max_val_A << endl;
        cout << "与B模板的相似度：" << max_val_B << endl;
        // 保存中间结果，便于调试
        imwrite("red_mask.jpg", red_mask);       // 红色掩膜
        imwrite("warped_letter.jpg", warped_letter); // 矫正后的字母
        imwrite("template_A_used.png", template_A);   // 所用A模板
    }

    // 步骤5.4：判断结果（设置相似度阈值0.6，避免误判）
    const double match_threshold = 0.6;
    if (max_val_A > max_val_B && max_val_A > match_threshold) {
        return 1; // 识别为A
    } else if (max_val_B > max_val_A && max_val_B > match_threshold) {
        return 2; // 识别为B
    } else {
        if (debug) cout << "相似度低于阈值，无法识别" << endl;
        return 0; // 未识别到
    }
}

// AB识别
bool compareContourLeftmostX(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        cv::Rect bbox1 = cv::boundingRect(contour1);
        cv::Rect bbox2 = cv::boundingRect(contour2);
        return bbox1.x < bbox2.x;
}
int AB_detect(const Mat& frame_hsv) {
    Config::AB_x = 0;
    Config::AB_y = 0;
    // 生成蓝色掩码
    Mat blue_mask;
    inRange(frame_hsv, Config::lower_blue, Config::upper_blue, blue_mask);
    blue_mask = blue_mask(Range(130, 180), Range(40, 280));

    std::vector<std::vector<Point>> contours;
    findContours(blue_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.size() == 1) {
        State::have_one_blue = 1;
        Rect bounding_rect = boundingRect(contours[0]);
        int x = bounding_rect.x;
        int y = bounding_rect.y;
        int w = bounding_rect.width;
        int h = bounding_rect.height;
        Config::AB_x = 40 + x + w / 2;
        Config::AB_y = 120 + y + h / 2;
        return 1;
    }
    if (contours.size() >= 2) {
        // 计算两个轮廓的边界框
        Rect boundingBox1 = boundingRect(contours[0]);
        Rect boundingBox2 = boundingRect(contours[1]);

        // 获取左上角的 y 坐标
        int y1 = boundingBox1.y; // 第一个轮廓的左上角 y 坐标
        int y2 = boundingBox2.y; // 第二个轮廓的左上角 y 坐标

        // 计算 y 坐标差,并判断条件
        int y_diff = abs(y1 - y2);
        if (y_diff < 10 && contours.size() == 2) State::have_two_blue = 1;

        // 获取最左边的轮廓
        std::vector<cv::Point> leftMostContour;
        for (const auto& contour : contours) {
            if (leftMostContour.empty() || compareContourLeftmostX(contour, leftMostContour)) {
                leftMostContour = contour;
            }
        }
        // 查找轮廓并获取蓝色区域中心点坐标
        Rect bounding_rect = boundingRect(leftMostContour);
        int x = bounding_rect.x;
        int y = bounding_rect.y;
        int w = bounding_rect.width;
        int h = bounding_rect.height;
        Config::AB_x = 40 + x + w / 2;
        Config::AB_y = 120 + y + h / 2;
        return 1;
    }
    return 0;
}


// AB停车处理
void AB_stop(int AB_x, int AB_y){
	   Config::error = 0;

     if(1){ //右停
         for (int i = 0; i < 10; i++) {
             //Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
             Config::draw_left_line_x[i] += 40;
             Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
             Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
         }
     }else{ //左停
         for (int i = 0; i < 10; i++) {
             //Config::draw_right_line_x[i] = ((Config::down_scan_wide - i * Config::y_scan_wide) - b) / k;
             Config::draw_left_line_x[i] += 40;
             Config::draw_middle_line_x[i] = (Config::draw_left_line_x[i] + Config::draw_right_line_x[i]) / 2;
             Config::error += Config::draw_middle_line_x[i] - Config::x_middle;
         }
     }
     Config::error /= Config::all_count;
     
}
/************************************
有关锥桶转向
************************************/
//转向角度计算
int angleToPwm(int angle) {
    // 限制角度在有效范围内，防止舵机过载
    if (angle < SERVO_MIN) angle = SERVO_MIN;
    if (angle > SERVO_MAX) angle = SERVO_MAX;
    
    // 公式推导：0.5ms对应6，2.5ms对应32，线性映射角度到PWM
    // 角度每增加1°，PWM增加 (32-6)/180 ≈ 0.144
    return 58 + (int)(angle * 0.633);  // 通用公式，可根据舵机手册微调
}



// 提取黄色锥桶中心点
vector<Point2f> getConeCenters(Mat frame) {
    Mat hsv, mask, kernel;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, Config::lower_yellow, Config::upper_yellow, mask);

    kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point2f> centers;
    for (auto &cnt : contours) {
        if (contourArea(cnt) > 50) {
            Moments m = moments(cnt);
            centers.emplace_back(m.m10/m.m00, m.m01/m.m00);
        }
    }
    //if (!centers.empty()) {  // 确保向量非空，避免越界
    //    centers.erase(centers.begin());  // 删除第一个元素（索引0）
    //}
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
/**************************************************************************************** */


int main(){

    int program_run_count = 0;
    
    LaneChangeState lane_state = {0, {}, {}, {}};

    int AB_stop_time = 0;
    // 电机速度参数（展示图像时车停止）
    int motor_speed = 207;// 191 -206，原210
    int motor_cone_speed = 200;  // 过锥桶速度适当减速
    int motor_ABstop_speed = 202;

    // 添加帧率控制变量
    auto last_time = std::chrono::high_resolution_clock::now();
    const double target_fps = 30.0;  // 目标帧率30FPS

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

    cv::VideoCapture cap(2);
    
    waiting(cap);
    
    while (true) {

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cap.release();
            cap.open(-1);
            cap >> frame;
            continue;
        }

        if (!running) {
            gpioPWM(motor_pin, 128);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            pthread_join(threadId, nullptr);
            exit(0);
        }

        // 图像处理
        cv::Mat frame_hsv, frame_final, frame_red;
        std::tie(frame_hsv, frame, frame_final, frame_red) = image_process(frame);
        
        //AB识别测试
        //std::string result2 = simpleDetectAB(frame);
        //std::cout << "快速检测结果: " << result2 << std::endl;
        //int letter_result = letter_detect(frame, true);
        //std::cout<<"db检测"<<letter_result<<std::endl;

        // 巡线
        //if(State::yellow_cone_start==0||State::yellow_cone_end_2==1){
        line_walking(frame_final);
        // 斑马线检测+-
        // 修改检测条件，使其在适当的时候检测斑马线
        // 添加自动检测机制，不仅依赖cone_count，也根据距离和时间判断
        static int frame_counter = 0;
        frame_counter++;
        
        // 在特定条件下自动检测斑马线
        if ((Config::cone_count >= 1 && Config::cone_count <= 2) || 
            (frame_counter % 30 == 0)) {  // 每30帧检测一次，确保不会错过
           sidewalk_detect(frame_red, cap);
        }

        // 转向处理
        if (State::sidewalk_end_flag && State::turn_end_flag == 0) {
            //turn_process();
            // 转向时减速，提高稳定性
            gpioPWM(motor_pin, motor_speed);  // 假设motor_low_speed是低速值
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 减速缓冲
            // 平滑转向到目标角度（比如中等角度，而非极端值）
            int target_angle = 140;  // 假设需要转向30度
            gpioPWM(servo_pin, angleToPwm(target_angle));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 保持转向角度，完成转向
            // 转回中间位置（或根据路径需求调整）
            gpioPWM(servo_pin, servo_middle);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            gpioPWM(servo_pin, angleToPwm(180-target_angle));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 保持转向角度，完成转向
            // 转回中间位置（或根据路径需求调整）
            gpioPWM(servo_pin, servo_middle);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // 恢复电机速度，标记转向结束
            gpioPWM(motor_pin, motor_speed);
            Config::turn_time += 1; 
            State::turn_end_flag = 1;
        }
        // 蓝色锥桶处理
        if (Config::cone_count < 1) {
           cone_process(frame_hsv);
        }
        //黄色锥桶第一次处理
        if(State::yellow_cone_end == 0 && State::turn_end_flag == 1){
            State::yellow_cone_start=1;
            //while(Config::yellow_cone_k==0){
            int half_height = frame.rows/2 ;  // 下半部分起始行（图像高度的一半）
            // 截取区域：从(0, half_height)到(frame.cols, frame.rows)，即下半部分
            cv::Rect lower_roi(0, half_height, frame.cols, frame.rows - half_height);
            cv::Mat frame_lower = frame(lower_roi);  // 下半部分图像
            vector<Point2f> centers = getConeCenters(frame);
            if (fitLineToCones(centers, Config::yellow_cone_k, Config::yellow_cone_b)) {
            // 绘制拟合直线（延长为引导左边线，覆盖图像上下边界）
                //if(Config::yellow_cone_k!=0){
                Point p1(0, Config::yellow_cone_b);
                Point p2(frame.cols, Config::yellow_cone_k*frame.cols + Config::yellow_cone_b);
                line(frame, p1, p2, Scalar(0, 255, 0), 2);

                // 绘制锥桶中心点
                for (auto &p : centers) circle(frame, p, 3, Scalar(0, 0, 255), -1);
                int targetAngle = calculateTargetAngle(Config::yellow_cone_k);
                std::cout<<targetAngle<<"&&&&&"<<Config::yellow_cone_k<<"********************************************************"<<std::endl;
                //gpioPWM(motor_pin, 129); 
                //std::this_thread::sleep_for(std::chrono::milliseconds(300000)); 
                gpioPWM(motor_pin, motor_speed); 
                std::this_thread::sleep_for(std::chrono::milliseconds(300)); 
                gpioPWM(servo_pin, angleToPwm(targetAngle));
                //转向延时
                std::this_thread::sleep_for(std::chrono::milliseconds(500));// 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                gpioPWM(servo_pin, angleToPwm(180-targetAngle));
                std::this_thread::sleep_for(std::chrono::milliseconds(300));  // 保持转向角度，完成转向
                // 转回中间位置（或根据路径需求调整）
                gpioPWM(servo_pin, servo_middle);
                //std::this_thread::sleep_for(std::chrono::milliseconds(200));
                //gpioPWM(motor_pin, 201);
                State::yellow_cone_end=1;
                //}
            }
            //}
        }
        // AB区域停车
        if (State::yellow_cone_end_2 == 1 && State::yellow_cone_end==1 ){
          AB_detect(frame_hsv);
	        if(Config::AB_x != 0){
		         State::AB_flag = 1;
             std::cout << "AB_flag" << std::endl;
          }
          if(State::AB_flag == 1){
            //angle_outmax = 3;
            //angle_outmin = -3;
            AB_stop(Config::AB_x, Config::AB_y);
            std::cout << "stop" << std::endl;
          }
        }

        double error_final = pid_control(Config::error / 3);
        std::cout << error_final << std::endl;
        //std::cout << "sidewalk:" << State::sidewalk << std::endl;
        //if(Config::cone_count==0){
            if(program_run_count == 10) gpioPWM(servo_pin, servo_middle - error_final);
            if(Config::cone_x == 0 && program_run_count == 10 && State::AB_flag == 0) gpioPWM(motor_pin, motor_speed);
            if(State::AB_flag == 1) gpioPWM(motor_pin, motor_ABstop_speed);
            Config::count_compare++;
            program_run_count ++;
        
            if(program_run_count >= 10)  program_run_count = 10;
        //}

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
        if (cv::waitKey(1) == 27) {  // 按下 ESC 退出
            break;
        }
    }
    cap.release();
    gpioTerminate();
    system("sudo killall pigpiod");
}
