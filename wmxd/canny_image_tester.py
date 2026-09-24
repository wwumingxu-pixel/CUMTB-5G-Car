import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
DEFAULT_ROI_TOP = 0.32
DEFAULT_ROI_BOTTOM = 0.80
DEFAULT_LOW = 40
DEFAULT_HIGH = 60
FILTER_MEDIAN_GAUSSIAN = "中值→高斯"
FILTER_GAUSSIAN_MEDIAN = "高斯→中值"
FILTER_MEDIAN_ONLY = "仅中值滤波"
FILTER_GAUSSIAN_ONLY = "仅高斯滤波"
FILTER_MODES = (
    FILTER_MEDIAN_GAUSSIAN,
    FILTER_GAUSSIAN_MEDIAN,
    FILTER_MEDIAN_ONLY,
    FILTER_GAUSSIAN_ONLY,
)

DEFAULT_FILTER = FILTER_MEDIAN_ONLY
DEFAULT_KERNEL = 5
DEFAULT_SIGMA = 0.5


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def process_canny(image, roi_top, roi_bottom, low_threshold, high_threshold,
                  filter_mode, kernel_size, gaussian_sigma):
    resized = cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    top = max(0, min(PROCESS_HEIGHT - 1, int(PROCESS_HEIGHT * roi_top)))
    bottom = max(top + 1, min(PROCESS_HEIGHT, int(PROCESS_HEIGHT * roi_bottom)))
    roi_gray = gray[top:bottom, :]

    if filter_mode == FILTER_MEDIAN_GAUSSIAN:
        median = cv2.medianBlur(roi_gray, kernel_size)
        roi_filtered = cv2.GaussianBlur(
            median, (kernel_size, kernel_size), gaussian_sigma, gaussian_sigma
        )
        filter_label = (
            f"Median→Gaussian {kernel_size}x{kernel_size}, "
            f"sigma={gaussian_sigma:g}"
        )
    elif filter_mode == FILTER_GAUSSIAN_MEDIAN:
        gaussian = cv2.GaussianBlur(
            roi_gray, (kernel_size, kernel_size), gaussian_sigma, gaussian_sigma
        )
        roi_filtered = cv2.medianBlur(gaussian, kernel_size)
        filter_label = (
            f"Gaussian→Median {kernel_size}x{kernel_size}, "
            f"sigma={gaussian_sigma:g}"
        )
    elif filter_mode == FILTER_GAUSSIAN_ONLY:
        roi_filtered = cv2.GaussianBlur(
            roi_gray, (kernel_size, kernel_size), gaussian_sigma, gaussian_sigma
        )
        filter_label = (
            f"Gaussian {kernel_size}x{kernel_size}, sigma={gaussian_sigma:g}"
        )
    else:
        roi_filtered = cv2.medianBlur(roi_gray, kernel_size)
        filter_label = f"Median {kernel_size}x{kernel_size}"

    roi_edges = cv2.Canny(roi_filtered, low_threshold, high_threshold, apertureSize=3)

    filtered = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
    filtered[top:bottom, :] = roi_filtered
    edges = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
    edges[top:bottom, :] = roi_edges

    overlay = resized.copy()
    overlay[edges > 0] = (0, 0, 255)
    cv2.rectangle(overlay, (0, top), (PROCESS_WIDTH - 1, bottom - 1),
                  (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"{filter_label}  Canny {low_threshold}/{high_threshold}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return resized, filtered, edges, overlay, (top, bottom), filter_label


class CannyTester(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Canny ROI Image Tester")
        self.geometry("1180x680")
        self.minsize(900, 560)

        self.source_image = None
        self.resized = None
        self.filtered = None
        self.edges = None
        self.overlay = None
        self.roi = None
        self.left_photo = None
        self.right_photo = None
        self.display_mode = tk.StringVar(value="overlay")
        self.filter_var = tk.StringVar(value=DEFAULT_FILTER)
        self.kernel_var = tk.StringVar(value=str(DEFAULT_KERNEL))
        self.sigma_var = tk.StringVar(value=str(DEFAULT_SIGMA))
        self.low_var = tk.StringVar(value=str(DEFAULT_LOW))
        self.high_var = tk.StringVar(value=str(DEFAULT_HIGH))
        self.top_var = tk.StringVar(value=str(int(DEFAULT_ROI_TOP * 100)))
        self.bottom_var = tk.StringVar(value=str(int(DEFAULT_ROI_BOTTOM * 100)))
        self.status_text = tk.StringVar(value="请选择图片")
        self.sigma_entry = None
        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择图片", command=self.choose_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="选择区域", command=self.select_roi).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="取消区域", command=self.reset_roi).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(toolbar, text="滤波方式").pack(side=tk.LEFT, padx=(16, 4))
        filter_box = ttk.Combobox(toolbar, textvariable=self.filter_var, width=12,
                                  values=FILTER_MODES, state="readonly")
        filter_box.pack(side=tk.LEFT)
        filter_box.bind("<<ComboboxSelected>>", self.on_filter_mode_changed)
        ttk.Label(toolbar, text="核").pack(side=tk.LEFT, padx=(8, 4))
        kernel_box = ttk.Combobox(toolbar, textvariable=self.kernel_var, width=3,
                                  values=("3", "5", "7", "9"), state="readonly")
        kernel_box.pack(side=tk.LEFT)
        kernel_box.bind("<<ComboboxSelected>>", lambda _event: self.reprocess())
        ttk.Label(toolbar, text="Sigma").pack(side=tk.LEFT, padx=(8, 4))
        self.sigma_entry = ttk.Entry(toolbar, textvariable=self.sigma_var, width=4)
        self.sigma_entry.pack(side=tk.LEFT)
        self.update_filter_controls()

        ttk.Label(toolbar, text="低阈值").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(toolbar, textvariable=self.low_var, width=5).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="高阈值").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(toolbar, textvariable=self.high_var, width=5).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="ROI上").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(toolbar, textvariable=self.top_var, width=5).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="ROI下").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(toolbar, textvariable=self.bottom_var, width=5).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="应用参数", command=self.reprocess).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Radiobutton(toolbar, text="滤波图", value="filtered",
                        variable=self.display_mode,
                        command=self.refresh_right_image).pack(side=tk.LEFT, padx=(16, 4))
        ttk.Radiobutton(toolbar, text="Canny", value="edges",
                        variable=self.display_mode,
                        command=self.refresh_right_image).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(toolbar, text="叠加图", value="overlay",
                        variable=self.display_mode,
                        command=self.refresh_right_image).pack(side=tk.LEFT, padx=4)
        ttk.Label(toolbar, textvariable=self.status_text).pack(side=tk.RIGHT)

        content = ttk.Frame(self, padding=(10, 0, 10, 10))
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, text="原图").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(content, text="Canny结果").grid(row=0, column=1, sticky=tk.W,
                                                   padx=(10, 0), pady=(0, 5))
        self.left_label = ttk.Label(content, anchor=tk.CENTER, relief=tk.SOLID)
        self.right_label = ttk.Label(content, anchor=tk.CENTER, relief=tk.SOLID)
        self.left_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.right_label.grid(row=1, column=1, sticky=tk.NSEW, padx=(10, 0))
        self.bind("<Configure>", self.on_resize)

    def on_filter_mode_changed(self, _event=None):
        self.update_filter_controls()
        self.reprocess()

    def update_filter_controls(self):
        if self.sigma_entry is None:
            return
        uses_gaussian = self.filter_var.get() in (
            FILTER_MEDIAN_GAUSSIAN,
            FILTER_GAUSSIAN_MEDIAN,
            FILTER_GAUSSIAN_ONLY,
        )
        self.sigma_entry.configure(state=tk.NORMAL if uses_gaussian else tk.DISABLED)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择测试图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        image = read_image(path)
        if image is None:
            self.status_text.set("无法读取图片")
            return
        self.source_image = image
        self.roi = None
        self.process_current_image(os.path.basename(path))

    def select_roi(self):
        if self.source_image is None:
            self.status_text.set("请先选择图片")
            return
        title = "选择区域 - 拖动后按 Enter"
        x, y, width, height = cv2.selectROI(title, self.source_image,
                                            showCrosshair=True,
                                            fromCenter=False)
        cv2.destroyWindow(title)
        if width <= 0 or height <= 0:
            return
        self.roi = (int(x), int(y), int(width), int(height))
        self.process_current_image()

    def reset_roi(self):
        if self.source_image is not None:
            self.roi = None
            self.process_current_image()

    def reprocess(self):
        if self.source_image is not None:
            self.process_current_image()

    def process_current_image(self, filename=None):
        try:
            low = max(0, min(255, int(self.low_var.get())))
            high = max(low + 1, min(255, int(self.high_var.get())))
            top = max(0.0, min(0.99, float(self.top_var.get()) / 100.0))
            bottom = max(top + 0.01, min(1.0, float(self.bottom_var.get()) / 100.0))
            kernel = int(self.kernel_var.get())
            if kernel not in (3, 5, 7, 9):
                raise ValueError
            sigma = max(0.0, min(10.0, float(self.sigma_var.get())))
        except ValueError:
            self.status_text.set("参数格式错误")
            return

        image = self.source_image
        roi_name = "整图"
        if self.roi is not None:
            x, y, width, height = self.roi
            image = self.source_image[y:y + height, x:x + width].copy()
            roi_name = f"选区 {width}x{height}"

        self.resized, self.filtered, self.edges, self.overlay, roi_rows, filter_label = process_canny(
            image, top, bottom, low, high, self.filter_var.get(), kernel, sigma)
        name = filename if filename is not None else "当前图片"
        self.status_text.set(
            f"{name}   {roi_name}   输入={PROCESS_WIDTH}x{PROCESS_HEIGHT}   "
            f"ROI={roi_rows[0]}..{roi_rows[1]}   {filter_label}   Canny={low}/{high}"
        )
        self.refresh_images()

    def on_resize(self, _event):
        if self.source_image is not None:
            self.refresh_images()

    @staticmethod
    def to_photo(image, max_width, max_height):
        if image is None or max_width < 40 or max_height < 40:
            return None
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height)
        display = cv2.resize(image, (max(1, int(width * scale)),
                                     max(1, int(height * scale))),
                             interpolation=cv2.INTER_AREA)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def refresh_images(self):
        self.update_idletasks()
        left_width = self.left_label.winfo_width()
        left_height = self.left_label.winfo_height()
        right_width = self.right_label.winfo_width()
        right_height = self.right_label.winfo_height()

        left_image = self.source_image.copy()
        if self.roi is not None:
            x, y, width, height = self.roi
            cv2.rectangle(left_image, (x, y), (x + width, y + height),
                          (0, 255, 255), 3, cv2.LINE_AA)
        self.left_photo = self.to_photo(left_image, left_width - 12, left_height - 12)
        if self.left_photo is not None:
            self.left_label.configure(image=self.left_photo)
        self.refresh_right_image(right_width, right_height)

    def refresh_right_image(self, width=None, height=None):
        if self.filtered is None or self.edges is None or self.overlay is None:
            return
        if width is None:
            width = self.right_label.winfo_width()
        if height is None:
            height = self.right_label.winfo_height()
        mode = self.display_mode.get()
        image = self.filtered if mode == "filtered" else self.edges if mode == "edges" else self.overlay
        self.right_photo = self.to_photo(image, width - 12, height - 12)
        if self.right_photo is not None:
            self.right_label.configure(image=self.right_photo)


if __name__ == "__main__":
    CannyTester().mainloop()
