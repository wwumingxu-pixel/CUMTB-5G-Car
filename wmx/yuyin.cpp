// 音频播放测试程序
// 用法：
//   ./yuyin /home/pi/audio/abc.wav
//   ./yuyin /home/pi/audio/abc.wav hw:1,0
// 不传参数时，使用默认音频文件：/home/pi/abc.wav

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;

namespace AudioConfig {
const char* default_file = "/home/pi/abc.wav";
const char* default_device = "default";
}

int main(int argc, char* argv[]) {
    const string audio_file = argc >= 2
        ? argv[1]
        : AudioConfig::default_file;
    const string audio_device = argc >= 3
        ? argv[2]
        : AudioConfig::default_device;

    if (access(audio_file.c_str(), R_OK) != 0) {
        cerr << "无法读取音频文件: " << audio_file << endl;
        cerr << "错误: " << strerror(errno) << endl;
        return 1;
    }

    cout << "开始播放: " << audio_file << endl;
    cout << "音频设备: " << audio_device << endl;

    pid_t child = fork();
    if (child < 0) {
        cerr << "创建播放进程失败: " << strerror(errno) << endl;
        return 1;
    }

    if (child == 0) {
        execlp(
            "aplay",
            "aplay",
            "-D",
            audio_device.c_str(),
            audio_file.c_str(),
            static_cast<char*>(nullptr)
        );

        cerr << "启动 aplay 失败: " << strerror(errno) << endl;
        _exit(127);
    }

    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        cerr << "等待播放结束失败: " << strerror(errno) << endl;
        return 1;
    }

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        cout << "播放完成。" << endl;
        return 0;
    }

    if (WIFEXITED(status)) {
        cerr << "播放失败，aplay 返回码: " << WEXITSTATUS(status) << endl;
    } else if (WIFSIGNALED(status)) {
        cerr << "播放进程被信号终止: " << WTERMSIG(status) << endl;
    } else {
        cerr << "播放失败。" << endl;
    }

    return 1;
}
