# Canny 视觉测试工具

本目录包含三个 Windows 测试工具，以及在 EXE 无法启动时使用的源代码和依赖说明。

## 工具

| 文件 | 功能 |
| --- | --- |
| `dist/CannyTester.exe` | 读取图片，调整滤波、Canny 阈值和 ROI，查看边缘及叠加效果。 |
| `dist/CannyVideoTester.exe` | 读取 MP4、AVI、MOV、MKV、WEBM 等视频，逐帧测试 Canny、霍夫线和巡线效果。 |
| `dist/CannyFillScanTester.exe` | 在视频测试基础上增加双重滤波、外侧填白和中心扫描测试。 |

三个 EXE 是 PyInstaller 单文件程序，OpenCV、NumPy、Pillow、Tkinter 运行库已经打包进 EXE，正常情况下不需要另外复制 DLL。首次启动时 Windows 可能需要等待几秒解压临时运行文件。

## 使用方式

最简单的方式是双击 `run_canny_tools.bat`，选择要运行的工具。也可以直接双击 `dist` 目录中的 EXE。

工具启动后：

- `CannyTester` 点击“选择图片”，选择仓库 `testimage` 中的 JPG、PNG 等图片。
- `CannyVideoTester` 和 `CannyFillScanTester` 点击“选择视频”，选择 MP4、AVI、MOV、MKV 或 WEBM 文件。
- 修改参数后点击“应用参数”，可以观察不同滤波核、阈值和 ROI 对结果的影响。

## 如果 EXE 双击没有反应

GUI 程序不会把错误窗口显示在终端中，请按下面顺序排查：

1. 将整个 `CUMTB-5G-Car-v1.2.0-Windows` 文件夹解压到本地磁盘，不要直接在压缩包内运行。
2. 将文件夹加入 Windows Defender 或杀毒软件的信任范围。PyInstaller 单文件程序首次运行会解压文件，可能被安全软件拦截。
3. 右键 EXE，选择“属性”，如果底部有“解除锁定”，勾选后应用。
4. 在 CMD 中运行 EXE，查看错误信息：

```bat
cd /d C:\path\to\CUMTB-5G-Car-v1.2.0-Windows
wmxd\dist\CannyTester.exe
```

5. 如果仍然无法运行，使用源码方式运行。先安装 64 位 Python 3.10 或更高版本，然后在 `wmxd` 目录执行：

```bat
py -m pip install -r requirements.txt
py canny_image_tester.py
py canny_video_tester.py
py canny_fill_scan_tester.py
```

官方 Python Windows 安装包需要启用 Tkinter；不要使用精简版或仅 embeddable 版本的 Python。

## 文件关系

- `canny_image_tester.py`：`CannyTester.exe` 对应源码。
- `canny_video_tester.py`：`CannyVideoTester.exe` 对应源码，也是 `CannyFillScanTester.exe` 的基础模块。
- `canny_fill_scan_tester.py`：`CannyFillScanTester.exe` 对应源码。
- `Canny*.spec`：PyInstaller 打包配置，可用于重新构建 EXE。
- `requirements.txt`：源码运行和重新打包所需的第三方 Python 依赖。
- `run_canny_tools.bat`：工具选择启动脚本。

这些工具只进行图像和视频处理，不控制树莓派 GPIO、电机或舵机。