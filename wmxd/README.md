# Canny 视觉测试工具

本目录提供 Windows 下运行的三个视觉算法测试工具，用于离线验证 Canny 边缘检测、巡线和图像处理效果。

## 工具说明

| 工具 | 说明 |
| --- | --- |
| `CannyFillScanTester.exe` | Canny 边缘检测结合填充/扫描方式进行巡线测试，适合观察边线填充和扫描结果。 |
| `CannyTester.exe` | 对测试图片进行 Canny 边缘检测和巡线效果验证，适合快速比较参数。 |
| `CannyVideoTester.exe` | 读取视频或摄像头画面进行 Canny 巡线测试，适合观察连续帧效果。 |

## 使用方法

1. 下载对应的 `.exe` 文件。
2. 双击运行，或在 PowerShell / CMD 中启动。
3. 根据程序提示选择测试图片、视频或摄像头设备。
4. 测试图片和视频建议使用仓库中的 [`testimage`](../testimage/) 目录中的素材。
5. 若程序被 Windows Defender 或杀毒软件拦截，请确认文件来源后再允许运行。

这些工具是 Windows 打包版本，主要用于算法验证，不直接控制树莓派 GPIO、电机或舵机。实际参数效果会受到图片分辨率、摄像头视角、光照和赛道材质影响。

## 版本说明

本工具包随 `CUMTB-5G-Car` Release 发布。工具与源码中的 Canny/巡线测试逻辑对应，但 Windows 可执行文件不包含 Linux 下的 OpenCV、pigpio 或树莓派运行环境。
