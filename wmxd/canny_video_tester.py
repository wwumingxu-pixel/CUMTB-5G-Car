import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
HOUGH_MIN_LINE_LENGTH = 60


class CannyVideoTester(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Canny Hough Video Tester")
        self.geometry("1370x710")
        self.minsize(1000, 580)

        self.capture = None
        self.video_path = ""
        self.video_fps = 25.0
        self.playing = False
        self.timer_id = None
        self.source = None
        self.edges = None
        self.debug = None
        self.photos = {}
        self.pid_last_error = 0.0
        self.pid_previous_error = 0.0
        self.pid_output = 0.0

        self.filter_var = tk.StringVar(value="gaussian")
        self.kernel_var = tk.StringVar(value="5")
        self.sigma_var = tk.StringVar(value="0.5")
        self.low_var = tk.StringVar(value="40")
        self.high_var = tk.StringVar(value="60")
        self.top_var = tk.StringVar(value="25")
        self.bottom_var = tk.StringVar(value="85")
        self.status_var = tk.StringVar(value="选择视频后开始逐帧测试")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择视频", command=self.choose_video).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="播放", command=self.play).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="暂停", command=self.pause).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="单帧", command=self.step).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="重新开始", command=self.restart).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        settings = ttk.LabelFrame(self, text="处理参数", padding=(10, 8))
        settings.pack(fill=tk.X, padx=10)
        ttk.Label(settings, text="滤波").grid(row=0, column=0, sticky=tk.W)
        mode = ttk.Combobox(settings, textvariable=self.filter_var, width=9,
                            values=("gaussian", "median"), state="readonly")
        mode.grid(row=0, column=1, padx=(4, 12))
        mode.bind("<<ComboboxSelected>>", lambda _event: self.reprocess())
        self._entry(settings, 0, 2, "核", self.kernel_var, 4)
        self._entry(settings, 0, 4, "Sigma", self.sigma_var, 5)
        self._entry(settings, 0, 6, "Canny低", self.low_var, 4)
        self._entry(settings, 0, 8, "Canny高", self.high_var, 4)
        self._entry(settings, 0, 10, "ROI上%", self.top_var, 4)
        self._entry(settings, 0, 12, "ROI下%", self.bottom_var, 4)
        ttk.Button(settings, text="应用参数", command=self.reprocess).grid(
            row=0, column=14, padx=(10, 0)
        )
        ttk.Label(settings, text="中值滤波时 Sigma 不参与计算").grid(
            row=1, column=0, columnspan=15, sticky=tk.W, pady=(6, 0)
        )

        content = ttk.Frame(self, padding=(10, 10, 10, 10))
        content.pack(fill=tk.BOTH, expand=True)
        for column in range(3):
            content.columnconfigure(column, weight=1)
        content.rowconfigure(1, weight=1)

        self.labels = {}
        for column, (key, title) in enumerate((
            ("source", "原图与 ROI"),
            ("edges", "Canny 边缘"),
            ("debug", "霍夫线、中线与补线"),
        )):
            ttk.Label(content, text=title).grid(row=0, column=column,
                                                 sticky=tk.W, padx=(0 if column == 0 else 8, 0),
                                                 pady=(0, 5))
            label = ttk.Label(content, anchor=tk.CENTER, relief=tk.SOLID)
            label.grid(row=1, column=column, sticky=tk.NSEW,
                       padx=(0 if column == 0 else 8, 0))
            self.labels[key] = label

    @staticmethod
    def _entry(parent, row, column, title, variable, width):
        ttk.Label(parent, text=title).grid(row=row, column=column,
                                            sticky=tk.W, padx=(0, 4))
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=column + 1, padx=(0, 10)
        )

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="选择测试视频",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self.pause()
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.capture = None
            self.status_var.set("无法打开视频")
            return
        self.video_path = path
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps < 1.0:
            self.video_fps = 25.0
        self.status_var.set(f"{os.path.basename(path)}  {self.video_fps:.1f} fps")
        self.play()

    def _parameters(self):
        kernel = int(self.kernel_var.get())
        if kernel not in (3, 5, 7, 9):
            raise ValueError("滤波核只能为 3、5、7、9")
        low = max(0, min(255, int(self.low_var.get())))
        high = max(low + 1, min(255, int(self.high_var.get())))
        sigma = max(0.0, min(10.0, float(self.sigma_var.get())))
        top = max(0.0, min(0.99, float(self.top_var.get()) / 100.0))
        bottom = max(top + 0.01, min(1.0, float(self.bottom_var.get()) / 100.0))
        return kernel, low, high, sigma, top, bottom

    @staticmethod
    def _x_at_y(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return float(x1)
        return x1 + (y - y1) * (x2 - x1) / float(y2 - y1)

    def _reset_pid(self):
        self.pid_last_error = 0.0
        self.pid_previous_error = 0.0
        self.pid_output = 0.0

    def _pid_update(self, error):
        # Match canny.cpp: kp=0.25, ki=0, kd=0.05,
        # output limit=100 and per-frame increment limit=20.
        delta = (0.25 * (error - self.pid_last_error) +
                 0.05 * (error - 2.0 * self.pid_last_error + self.pid_previous_error))
        delta = max(-20.0, min(20.0, delta))
        self.pid_output = max(-100.0, min(100.0, self.pid_output + delta))
        self.pid_previous_error = self.pid_last_error
        self.pid_last_error = error
        return self.pid_output

    def _process_frame(self, frame):
        kernel, low, high, sigma, top_ratio, bottom_ratio = self._parameters()
        resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        top = int(PROCESS_HEIGHT * top_ratio)
        bottom = max(top + 1, int(PROCESS_HEIGHT * bottom_ratio))
        bottom = min(PROCESS_HEIGHT, bottom)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[top:bottom, :]
        if self.filter_var.get() == "median":
            filtered = cv2.medianBlur(roi_gray, kernel)
            filter_name = f"Median {kernel}x{kernel}"
        else:
            filtered = cv2.GaussianBlur(roi_gray, (kernel, kernel), sigma, sigma)
            filter_name = f"Gaussian {kernel}x{kernel}, sigma={sigma:g}"
        roi_edges = cv2.Canny(filtered, low, high, apertureSize=3)

        edge_view = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
        edge_view[top:bottom, :] = roi_edges
        line_view = np.zeros((bottom - top, PROCESS_WIDTH, 3), dtype=np.uint8)
        debug_roi = cv2.cvtColor(resized[top:bottom, :], cv2.COLOR_BGR2RGB)
        debug_roi = cv2.cvtColor(debug_roi, cv2.COLOR_RGB2BGR)

        lines = cv2.HoughLinesP(roi_edges, 1.0, np.pi / 180.0,
                                 threshold=20, minLineLength=HOUGH_MIN_LINE_LENGTH,
                                 maxLineGap=35)
        image_center = PROCESS_WIDTH / 2.0
        control_y = int(0.80 * (roi_edges.shape[0] - 1))
        left = None
        right = None
        if lines is not None:
            for item in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(value) for value in item]
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dx, dy = x2 - x1, y2 - y1
                length = float(np.hypot(dx, dy))
                if length < HOUGH_MIN_LINE_LENGTH or dy < 5.0:
                    continue
                tilt = np.degrees(np.arctan2(abs(dx), dy))
                if tilt < 15.0 or tilt > 75.0:
                    continue
                top_x = self._x_at_y((x1, y1, x2, y2), 0)
                bottom_x = self._x_at_y((x1, y1, x2, y2), roi_edges.shape[0] - 1)
                middle_x = (x1 + x2) * 0.5
                candidate = (length, (x1, y1, x2, y2))
                if dx < 0 and middle_x < image_center and top_x < image_center and bottom_x < image_center:
                    if left is None or length > left[0]:
                        left = candidate
                elif dx > 0 and middle_x > image_center and top_x > image_center and bottom_x > image_center:
                    if right is None or length > right[0]:
                        right = candidate

        left_fallback = left is None
        right_fallback = right is None
        if left_fallback:
            left = (0.0, (0, 0, 0, roi_edges.shape[0] - 1))
        if right_fallback:
            right = (0.0, (PROCESS_WIDTH - 1, 0,
                           PROCESS_WIDTH - 1, roi_edges.shape[0] - 1))

        def draw_side(candidate, left_side, fallback):
            _, line = candidate
            x1, y1, x2, y2 = line
            if fallback:
                cv2.line(line_view, (x1, y1), (x2, y2),
                         (0, 0, 255), 2, cv2.LINE_AA)
                return
            cv2.line(line_view, (x1, y1), (x2, y2), (150, 150, 150), 3, cv2.LINE_AA)
            if y2 < roi_edges.shape[0] - 1:
                corner_x = 0 if left_side else PROCESS_WIDTH - 1
                cv2.line(line_view, (x2, y2), (corner_x, roi_edges.shape[0] - 1),
                         (0, 0, 255), 2, cv2.LINE_AA)

        draw_side(left, True, left_fallback)
        draw_side(right, False, right_fallback)
        error_text = "error=NA pid=0"
        if left is not None and right is not None:
            left_x = self._x_at_y(left[1], control_y)
            right_x = self._x_at_y(right[1], control_y)
            if right_x > left_x:
                # Match canny.cpp: sample the lane center every four rows and
                # use a Gaussian centered at the control row for the final error.
                sample_step = 4
                sigma = max(8.0, roi_edges.shape[0] * 0.24)
                weighted_center_sum = 0.0
                weight_sum = 0.0
                center_samples = []
                for y in range(0, roi_edges.shape[0], sample_step):
                    sample_left_x = self._x_at_y(left[1], y)
                    sample_right_x = self._x_at_y(right[1], y)
                    if sample_right_x <= sample_left_x:
                        continue
                    center_x = (sample_left_x + sample_right_x) * 0.5
                    distance = y - control_y
                    weight = np.exp(-(distance * distance) / (2.0 * sigma * sigma))
                    weighted_center_sum += center_x * weight
                    weight_sum += weight
                    center_samples.append((int(round(center_x)), y))

                if weight_sum > 0.0:
                    center_x = weighted_center_sum / weight_sum
                    error = center_x - image_center
                    for index in range(1, len(center_samples)):
                        cv2.line(line_view, center_samples[index - 1], center_samples[index],
                                 (255, 255, 255), 2, cv2.LINE_AA)
                    pid = self._pid_update(error)
                    fill_state = ""
                    if left_fallback:
                        fill_state += " L-FILL"
                    if right_fallback:
                        fill_state += " R-FILL"
                    error_text = f"error={error:.1f} pid={pid:.1f}{fill_state}"
                else:
                    self._reset_pid()
            else:
                self._reset_pid()
        else:
            self._reset_pid()

        debug_roi = cv2.addWeighted(debug_roi, 0.65, line_view, 0.35, 0.0)
        cv2.line(debug_roi, (PROCESS_WIDTH // 2, 0),
                 (PROCESS_WIDTH // 2, debug_roi.shape[0] - 1), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(debug_roi, (0, control_y), (PROCESS_WIDTH - 1, control_y),
                 (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(debug_roi, error_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        source_view = resized.copy()
        cv2.rectangle(source_view, (0, top), (PROCESS_WIDTH - 1, bottom - 1),
                      (0, 255, 255), 1, cv2.LINE_AA)
        debug_view = np.zeros_like(resized)
        debug_view[top:bottom, :] = debug_roi
        self.source = source_view
        self.edges = cv2.cvtColor(edge_view, cv2.COLOR_GRAY2BGR)
        self.debug = debug_view
        frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) if self.capture else 0
        self.status_var.set(
            f"{os.path.basename(self.video_path)}  frame={frame_index}  "
            f"ROI={top}..{bottom}  {filter_name}  Canny={low}/{high}  {error_text}"
        )
        self._refresh_images()

    def play(self):
        if self.capture is None:
            return
        self.playing = True
        if self.timer_id is None:
            self._next_frame()

    def pause(self):
        self.playing = False
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None

    def step(self):
        self.pause()
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            try:
                self._process_frame(frame)
            except ValueError as error:
                self.status_var.set(str(error))
        else:
            self.status_var.set("视频播放结束")

    def restart(self):
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._reset_pid()
        self.play()

    def reprocess(self):
        if self.capture is None:
            return
        current = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, current)
        self.step()

    def _next_frame(self):
        self.timer_id = None
        if not self.playing or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.playing = False
            self.status_var.set("视频播放结束")
            return
        try:
            self._process_frame(frame)
        except ValueError as error:
            self.playing = False
            self.status_var.set(str(error))
            return
        delay = max(1, int(round(1000.0 / self.video_fps)))
        self.timer_id = self.after(delay, self._next_frame)

    @staticmethod
    def _to_photo(image, max_width, max_height):
        if image is None or max_width < 40 or max_height < 40:
            return None
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height)
        shown = cv2.resize(image, (max(1, int(width * scale)),
                                   max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _refresh_images(self):
        self.update_idletasks()
        for key, image in (("source", self.source), ("edges", self.edges), ("debug", self.debug)):
            label = self.labels[key]
            photo = self._to_photo(image, label.winfo_width() - 8, label.winfo_height() - 8)
            if photo is not None:
                self.photos[key] = photo
                label.configure(image=photo)

    def _on_resize(self, _event):
        if self.source is not None:
            self._refresh_images()

    def _close(self):
        self.pause()
        if self.capture is not None:
            self.capture.release()
        self.destroy()


if __name__ == "__main__":
    CannyVideoTester().mainloop()
