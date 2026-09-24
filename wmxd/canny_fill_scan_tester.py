import os
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from canny_video_tester import (
    CannyVideoTester,
    HOUGH_MIN_LINE_LENGTH,
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
)


SAME_SIDE_MAX_X_SPREAD = 28.0
FILTER_MEDIAN_GAUSSIAN = "中值 → 高斯"
FILTER_GAUSSIAN_MEDIAN = "高斯 → 中值"
FILTER_MEDIAN_ONLY = "仅中值滤波"
FILTER_GAUSSIAN_ONLY = "仅高斯滤波"


class CannyFillScanTester(CannyVideoTester):
    """Hough boundaries -> outside fill -> center-outward row scan."""

    def __init__(self):
        super().__init__()
        self.title("Canny 双重滤波与中心扫描测试器")
        self.geometry("1320x760")
        self.minsize(1040, 640)
        self.filter_var.set(FILTER_MEDIAN_GAUSSIAN)
        self.kernel_var.set("3")
        self.low_var.set("50")
        self.high_var.set("75")

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(12, 10, 12, 8))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择视频", command=self.choose_video).pack(side=tk.LEFT)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        ttk.Button(toolbar, text="播放", command=self.play).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="暂停", command=self.pause).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="单帧", command=self.step).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="重新开始", command=self.restart).pack(side=tk.LEFT, padx=(6, 0))

        settings = ttk.Frame(self, padding=(12, 0, 12, 8))
        settings.pack(fill=tk.X)
        settings.columnconfigure(0, weight=3)
        settings.columnconfigure(1, weight=2)
        settings.columnconfigure(2, weight=2)

        filter_group = ttk.LabelFrame(settings, text="双重滤波", padding=(10, 8))
        filter_group.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 8))
        ttk.Label(filter_group, text="处理顺序").grid(row=0, column=0, sticky=tk.W)
        mode = ttk.Combobox(
            filter_group,
            textvariable=self.filter_var,
            width=17,
            values=(
                FILTER_MEDIAN_GAUSSIAN,
                FILTER_GAUSSIAN_MEDIAN,
                FILTER_MEDIAN_ONLY,
                FILTER_GAUSSIAN_ONLY,
            ),
            state="readonly",
        )
        mode.grid(row=0, column=1, sticky=tk.W, padx=(6, 14))
        mode.bind("<<ComboboxSelected>>", lambda _event: self.reprocess())
        ttk.Label(filter_group, text="滤波核").grid(row=0, column=2, sticky=tk.W)
        kernel = ttk.Combobox(
            filter_group,
            textvariable=self.kernel_var,
            width=6,
            values=("3", "5", "7", "9"),
            state="readonly",
        )
        kernel.grid(row=0, column=3, sticky=tk.W, padx=(6, 14))
        kernel.bind("<<ComboboxSelected>>", lambda _event: self.reprocess())
        ttk.Label(filter_group, text="Sigma").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(filter_group, textvariable=self.sigma_var, width=7).grid(
            row=0, column=5, sticky=tk.W, padx=(6, 0)
        )

        canny_group = ttk.LabelFrame(settings, text="Canny 阈值", padding=(10, 8))
        canny_group.grid(row=0, column=1, sticky=tk.NSEW, padx=(0, 8))
        self._entry(canny_group, 0, 0, "低阈值", self.low_var, 6)
        self._entry(canny_group, 0, 2, "高阈值", self.high_var, 6)

        roi_group = ttk.LabelFrame(settings, text="ROI 范围", padding=(10, 8))
        roi_group.grid(row=0, column=2, sticky=tk.NSEW)
        self._entry(roi_group, 0, 0, "顶部 %", self.top_var, 6)
        self._entry(roi_group, 0, 2, "底部 %", self.bottom_var, 6)

        actions = ttk.Frame(self, padding=(12, 0, 12, 8))
        actions.pack(fill=tk.X)
        ttk.Label(actions, text="滤波核用于当前选择的单个或两个滤波器").pack(side=tk.LEFT)
        ttk.Button(actions, text="应用参数", command=self.reprocess).pack(side=tk.RIGHT)
        ttk.Button(actions, text="恢复默认", command=self.reset_defaults).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        content = ttk.Frame(self, padding=(12, 2, 12, 8))
        content.pack(fill=tk.BOTH, expand=True)
        for column in range(3):
            content.columnconfigure(column, weight=1, uniform="preview")
        content.rowconfigure(1, weight=1)

        self.labels = {}
        for column, (key, title) in enumerate((
            ("source", "原始画面 / ROI"),
            ("edges", "双重滤波后的 Canny 边缘"),
            ("debug", "霍夫拟合 / 外侧填白 / 中心扫描"),
        )):
            ttk.Label(content, text=title).grid(
                row=0,
                column=column,
                sticky=tk.W,
                padx=(0 if column == 0 else 8, 0),
                pady=(0, 5),
            )
            label = ttk.Label(content, anchor=tk.CENTER, relief=tk.SOLID)
            label.grid(
                row=1,
                column=column,
                sticky=tk.NSEW,
                padx=(0 if column == 0 else 8, 0),
            )
            self.labels[key] = label

        status = ttk.Frame(self, padding=(12, 5, 12, 7))
        status.pack(fill=tk.X)
        ttk.Separator(status).pack(fill=tk.X, pady=(0, 5))
        ttk.Label(status, text="状态：").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.status_var, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

    def reset_defaults(self):
        self.filter_var.set(FILTER_MEDIAN_GAUSSIAN)
        self.kernel_var.set("3")
        self.sigma_var.set("0.5")
        self.low_var.set("50")
        self.high_var.set("75")
        self.top_var.set("25")
        self.bottom_var.set("85")
        self.reprocess()

    @staticmethod
    def _fit_side_candidates(candidates):
        if not candidates:
            return None

        # Cluster by x at the control row. Every pair in the selected group is
        # therefore no farther apart than SAME_SIDE_MAX_X_SPREAD horizontally.
        ordered = sorted(candidates, key=lambda candidate: candidate[2])
        best_group = []
        best_score = (-1.0, -1)
        for start, first in enumerate(ordered):
            group = []
            support_length = 0.0
            for candidate in ordered[start:]:
                if candidate[2] - first[2] > SAME_SIDE_MAX_X_SPREAD:
                    break
                group.append(candidate)
                support_length += candidate[0]
                score = (support_length, len(group))
                if score > best_score:
                    best_score = score
                    best_group = list(group)

        support_lines = [candidate[1] for candidate in best_group]
        if len(support_lines) == 1:
            return best_group[0][0], support_lines[0], support_lines

        points = np.asarray(
            [(x1, y1) for x1, y1, _, _ in support_lines] +
            [(x2, y2) for _, _, x2, y2 in support_lines],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        vx, vy, x0, y0 = np.ravel(
            cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
        )
        if abs(float(vy)) < 1e-6:
            longest = max(best_group, key=lambda candidate: candidate[0])
            return best_score[0], longest[1], support_lines

        slope = float(vx) / float(vy)
        min_y = min(min(line[1], line[3]) for line in support_lines)
        max_y = max(max(line[1], line[3]) for line in support_lines)
        top_x = float(x0) + (min_y - float(y0)) * slope
        bottom_x = float(x0) + (max_y - float(y0)) * slope
        fitted_line = (
            int(round(top_x)), int(min_y),
            int(round(bottom_x)), int(max_y),
        )
        return best_score[0], fitted_line, support_lines

    @staticmethod
    def _select_hough_sides(edges):
        lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0,
                                 threshold=20, minLineLength=HOUGH_MIN_LINE_LENGTH,
                                 maxLineGap=35)
        image_center = edges.shape[1] / 2.0
        bottom_y = edges.shape[0] - 1
        control_y = int(0.80 * bottom_y)
        left_candidates = []
        right_candidates = []
        if lines is None:
            return None, None

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
            if tilt < 5.0 or tilt > 85.0:
                continue

            line = (x1, y1, x2, y2)
            reference_x = CannyFillScanTester._x_at_y(line, control_y)
            candidate = (length, line, reference_x)
            # x1/y1 is the upper line start after endpoint normalization.
            if dx < 0 and x1 < image_center:
                left_candidates.append(candidate)
            elif dx > 0 and x1 > image_center:
                right_candidates.append(candidate)

        left = CannyFillScanTester._fit_side_candidates(left_candidates)
        right = CannyFillScanTester._fit_side_candidates(right_candidates)
        return left, right

    @staticmethod
    def _boundary_x_at_row(line, y, bottom_y, width, left_side):
        x1, y1, x2, y2 = line
        if y <= y2 or y2 >= bottom_y:
            return CannyFillScanTester._x_at_y(line, y)
        corner_x = 0 if left_side else width - 1
        ratio = (y - y2) / float(bottom_y - y2)
        return x2 + (corner_x - x2) * ratio

    def _build_scan_mask(self, height, width, left, right):
        mask = np.zeros((height, width), dtype=np.uint8)
        bottom_y = height - 1
        if left is not None:
            for y in range(height):
                boundary = self._boundary_x_at_row(left[1], y, bottom_y, width, True)
                boundary = int(np.clip(round(boundary), 0, width - 1))
                mask[y, :boundary + 1] = 255
        if right is not None:
            for y in range(height):
                boundary = self._boundary_x_at_row(right[1], y, bottom_y, width, False)
                boundary = int(np.clip(round(boundary), 0, width - 1))
                mask[y, boundary:] = 255
        return mask

    def _process_frame(self, frame):
        kernel, low, high, sigma, top_ratio, bottom_ratio = self._parameters()
        resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        top = int(PROCESS_HEIGHT * top_ratio)
        bottom = min(PROCESS_HEIGHT, max(top + 1, int(PROCESS_HEIGHT * bottom_ratio)))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[top:bottom, :]
        filter_mode = self.filter_var.get()
        if filter_mode == FILTER_GAUSSIAN_MEDIAN:
            gaussian = cv2.GaussianBlur(roi_gray, (kernel, kernel), sigma, sigma)
            filtered = cv2.medianBlur(gaussian, kernel)
            filter_name = f"Gaussian>Median {kernel}x{kernel}, sigma={sigma:g}"
        elif filter_mode == FILTER_MEDIAN_ONLY:
            filtered = cv2.medianBlur(roi_gray, kernel)
            filter_name = f"Median {kernel}x{kernel}"
        elif filter_mode == FILTER_GAUSSIAN_ONLY:
            filtered = cv2.GaussianBlur(roi_gray, (kernel, kernel), sigma, sigma)
            filter_name = f"Gaussian {kernel}x{kernel}, sigma={sigma:g}"
        else:
            median = cv2.medianBlur(roi_gray, kernel)
            filtered = cv2.GaussianBlur(median, (kernel, kernel), sigma, sigma)
            filter_name = f"Median>Gaussian {kernel}x{kernel}, sigma={sigma:g}"
        roi_edges = cv2.Canny(filtered, low, high, apertureSize=3)

        left, right = self._select_hough_sides(roi_edges)
        left_fallback = left is None
        right_fallback = right is None
        if left_fallback:
            left = (0.0, (0, 0, 0, roi_edges.shape[0] - 1), [])
        if right_fallback:
            right = (0.0, (PROCESS_WIDTH - 1, 0,
                           PROCESS_WIDTH - 1, roi_edges.shape[0] - 1), [])
        left_support_count = len(left[2])
        right_support_count = len(right[2])
        scan_mask = self._build_scan_mask(roi_edges.shape[0], PROCESS_WIDTH, left, right)
        debug_roi = cv2.cvtColor(scan_mask, cv2.COLOR_GRAY2BGR)
        bottom_y = roi_edges.shape[0] - 1

        def draw_hough_and_fill_line(candidate, left_side, fallback):
            _, (x1, y1, x2, y2), support_lines = candidate
            if fallback:
                cv2.line(debug_roi, (x1, y1), (x2, y2),
                         (0, 0, 255), 2, cv2.LINE_AA)
                return
            for sx1, sy1, sx2, sy2 in support_lines:
                cv2.line(debug_roi, (sx1, sy1), (sx2, sy2),
                         (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(debug_roi, (x1, y1), (x2, y2), (150, 150, 150), 3, cv2.LINE_AA)
            if y2 < bottom_y:
                corner_x = 0 if left_side else PROCESS_WIDTH - 1
                cv2.line(debug_roi, (x2, y2), (corner_x, bottom_y),
                         (0, 0, 255), 2, cv2.LINE_AA)

        draw_hough_and_fill_line(left, True, left_fallback)
        draw_hough_and_fill_line(right, False, right_fallback)

        control_y = int(0.80 * bottom_y)
        start_x = PROCESS_WIDTH // 2
        sample_step = 4
        sample_rows = list(range(bottom_y, -1, -sample_step))
        center_samples = []
        scan_pairs = []
        for y in sample_rows:
            start_x = int(np.clip(start_x, 0, PROCESS_WIDTH - 1))
            left_x = next((x for x in range(start_x, -1, -1) if scan_mask[y, x] != 0), None)
            right_x = next((x for x in range(start_x, PROCESS_WIDTH) if scan_mask[y, x] != 0), None)
            if left_x is None or right_x is None or right_x <= left_x:
                continue
            center_x = (left_x + right_x) * 0.5
            start_x = int(round(center_x))
            center_samples.append((center_x, y))
            scan_pairs.append((left_x, right_x, y))

        error_text = "error=NA pid=0"
        if center_samples:
            gaussian_sigma = max(8.0, roi_edges.shape[0] * 0.24)
            weighted_sum = 0.0
            weights = 0.0
            for center_x, y in center_samples:
                distance = y - control_y
                weight = np.exp(-(distance * distance) / (2.0 * gaussian_sigma * gaussian_sigma))
                weighted_sum += center_x * weight
                weights += weight
            measured_center = weighted_sum / weights
            error = measured_center - PROCESS_WIDTH / 2.0
            pid = self._pid_update(error)
            fill_state = ""
            if left_fallback:
                fill_state += " L-FILL"
            if right_fallback:
                fill_state += " R-FILL"
            error_text = f"error={error:.1f} pid={pid:.1f}{fill_state}"

            for left_x, right_x, y in scan_pairs:
                cv2.circle(debug_roi, (left_x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(debug_roi, (right_x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
            points = [(int(round(x)), y) for x, y in center_samples]
            for index in range(1, len(points)):
                cv2.line(debug_roi, points[index - 1], points[index],
                         (255, 255, 255), 2, cv2.LINE_AA)
        else:
            self._reset_pid()

        cv2.line(debug_roi, (PROCESS_WIDTH // 2, 0),
                 (PROCESS_WIDTH // 2, bottom_y), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(debug_roi, (0, control_y), (PROCESS_WIDTH - 1, control_y),
                 (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(debug_roi, error_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        source_view = resized.copy()
        cv2.rectangle(source_view, (0, top), (PROCESS_WIDTH - 1, bottom - 1),
                      (0, 255, 255), 1, cv2.LINE_AA)
        edge_view = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
        edge_view[top:bottom, :] = roi_edges
        debug_view = np.zeros_like(resized)
        debug_view[top:bottom, :] = debug_roi

        self.source = source_view
        self.edges = cv2.cvtColor(edge_view, cv2.COLOR_GRAY2BGR)
        self.debug = debug_view
        frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) if self.capture else 0
        self.status_var.set(
            f"{os.path.basename(self.video_path)}  frame={frame_index}  "
            f"ROI={top}..{bottom}  {filter_name}  Canny={low}/{high}  "
            f"Fit=L{left_support_count}/R{right_support_count}  {error_text}"
        )
        self._refresh_images()


if __name__ == "__main__":
    CannyFillScanTester().mainloop()
