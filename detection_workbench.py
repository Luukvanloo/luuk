"""Interactive OpenCV detection workbench for the project's images and videos."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DetectionWorkbench:
    DETECTIONS = {
        "Original": {},
        "Canny edge detection": {"low": (0, 255, 1, 50), "high": (1, 255, 1, 150)},
        "Sobel edge detection": {"direction": ("Both", "Horizontal", "Vertical"), "kernel": (1, 7, 2, 3)},
        "Laplacian edges": {"kernel": (1, 7, 2, 3), "threshold": (0, 255, 1, 30)},
        "Hough circle detection": {
            "dp": (10, 30, 1, 10), "min_distance": (5, 200, 1, 30),
            "param1": (1, 300, 1, 100), "param2": (1, 150, 1, 35),
            "min_radius": (0, 300, 1, 0), "max_radius": (0, 500, 1, 0),
        },
        "Hough line detection": {"threshold": (1, 300, 1, 100), "min_length": (1, 500, 1, 50), "max_gap": (1, 200, 1, 10)},
        "Contour detection": {"threshold": (0, 255, 1, 100), "min_area": (0, 100000, 100, 500)},
        "Corner detection": {"max_corners": (1, 500, 1, 100), "quality": (1, 100, 1, 10), "min_distance": (1, 100, 1, 10)},
        "Shi-Tomasi tracking points": {"max_corners": (1, 500, 1, 100), "quality": (1, 100, 1, 10), "min_distance": (1, 100, 1, 10)},
        "Blob detection": {"min_area": (1, 100000, 100, 100), "max_area": (100, 500000, 100, 50000), "circularity": (1, 100, 1, 10)},
        "Watershed segmentation": {"threshold": (0, 255, 1, 127)},
        "Foreground mask (MOG2)": {"history": (10, 1000, 10, 500), "variance": (1, 100, 1, 16)},
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenCV Detection Workbench")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)
        self.project_dir = Path(__file__).resolve().parent
        self.media_paths: list[Path] = []
        self.current_path: Path | None = None
        self.current_frame: np.ndarray | None = None
        self.video: cv2.VideoCapture | None = None
        self.video_playing = False
        self.video_fps = 25.0
        self.preview_photo: tk.PhotoImage | None = None
        self.parameter_vars: dict[str, tk.Variable] = {}
        self.parameter_widgets: list[tk.Widget] = []
        self.mog2: cv2.BackgroundSubtractor | None = None

        self.detection_var = tk.StringVar(value="Original")
        self.status_var = tk.StringVar(value="Choose a picture or video to begin.")
        self.play_var = tk.StringVar(value="Play")
        self._style()
        self._build_ui()
        self._load_media()

    def _style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#20252b")
        style.configure("Panel.TFrame", background="#2b3138")
        style.configure("TLabel", background="#2b3138", foreground="#f1f3f5")
        style.configure("Title.TLabel", background="#20252b", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", background="#20252b", foreground="#aab2bc")

    def _build_ui(self) -> None:
        ttk.Label(self.root, text="OpenCV Detection Workbench", style="Title.TLabel").pack(anchor="w", padx=18, pady=(14, 2))
        ttk.Label(self.root, text="Try common OpenCV edge, shape, feature, contour, segmentation, and motion detectors.", style="Muted.TLabel").pack(anchor="w", padx=18, pady=(0, 12))
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(left, text="Pictures and videos", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left, text="data/ and Broodjes_MP4/", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        self.source_list = tk.Listbox(left, width=30, height=20, bg="#171a1e", fg="#e8eaed", selectbackground="#3b82f6", borderwidth=0, highlightthickness=0, activestyle="none")
        self.source_list.pack(fill="both", expand=True)
        self.source_list.bind("<<ListboxSelect>>", self._source_selected)
        ttk.Button(left, text="Refresh sources", command=self._load_media).pack(fill="x", pady=(10, 4))
        ttk.Button(left, text="Open file...", command=self._open_file).pack(fill="x")

        right = ttk.Frame(main, style="Panel.TFrame", padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(right, style="Panel.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(toolbar, text="Detection:").pack(side="left")
        choice = ttk.Combobox(toolbar, textvariable=self.detection_var, values=list(self.DETECTIONS), state="readonly", width=30)
        choice.pack(side="left", padx=(8, 14))
        choice.bind("<<ComboboxSelected>>", self._detection_changed)
        ttk.Button(toolbar, textvariable=self.play_var, command=self._toggle_video).pack(side="left")
        ttk.Button(toolbar, text="Save result...", command=self._save_result).pack(side="right")
        self.preview = tk.Label(right, text="No media selected", bg="#111315", fg="#8d98a5")
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.parameters = ttk.Frame(right, style="Panel.TFrame")
        self.parameters.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(right, textvariable=self.status_var, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._rebuild_parameters()

    def _load_media(self) -> None:
        self.source_list.delete(0, tk.END)
        self.media_paths = []
        for directory, extensions in ((self.project_dir / "data", IMAGE_EXTENSIONS), (self.project_dir / "Broodjes_MP4", VIDEO_EXTENSIONS)):
            if directory.exists():
                self.media_paths.extend(sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions))
        for path in self.media_paths:
            self.source_list.insert(tk.END, str(path.relative_to(self.project_dir)))
        self.status_var.set(f"{len(self.media_paths)} media files found.")

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self._set_media(Path(path))

    def _source_selected(self, _event: object = None) -> None:
        selected = self.source_list.curselection()
        if selected:
            self._set_media(self.media_paths[selected[0]])

    def _set_media(self, path: Path) -> None:
        self._stop_video()
        self.current_path = path
        self.mog2 = None
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            self.video = cv2.VideoCapture(str(path))
            if not self.video.isOpened():
                messagebox.showerror("OpenCV", f"Could not open video:\n{path}")
                self.video = None
                return
            self.video_fps = self.video.get(cv2.CAP_PROP_FPS) or 25
            self._read_video_frame()
        else:
            self.current_frame = cv2.imread(str(path))
            if self.current_frame is None:
                messagebox.showerror("OpenCV", f"Could not read image:\n{path}")
                return
            self.status_var.set(f"Loaded {path.name}")
            self._refresh_preview()

    def _read_video_frame(self) -> None:
        if self.video is None:
            return
        success, frame = self.video.read()
        if success:
            self.current_frame = frame
            self._refresh_preview()
            if self.video_playing:
                self.root.after(max(1, round(1000 / self.video_fps)), self._read_video_frame)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_playing = False
            self.play_var.set("Play")

    def _toggle_video(self) -> None:
        if self.video is not None:
            self.video_playing = not self.video_playing
            self.play_var.set("Pause" if self.video_playing else "Play")
            if self.video_playing:
                self._read_video_frame()

    def _stop_video(self) -> None:
        self.video_playing = False
        if self.video is not None:
            self.video.release()
            self.video = None
        self.play_var.set("Play")

    def _detection_changed(self, _event: object = None) -> None:
        self._rebuild_parameters()
        self.mog2 = None
        self._refresh_preview()

    def _rebuild_parameters(self) -> None:
        for widget in self.parameter_widgets:
            widget.destroy()
        self.parameter_widgets.clear()
        self.parameter_vars.clear()
        for row, (name, definition) in enumerate(self.DETECTIONS[self.detection_var.get()].items()):
            ttk.Label(self.parameters, text=f"{name.replace('_', ' ').title()}:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            if isinstance(definition[0], str):
                variable = tk.StringVar(value=definition[0])
                widget = ttk.Combobox(self.parameters, textvariable=variable, values=definition, state="readonly", width=14)
                widget.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preview())
            else:
                minimum, maximum, step, default = definition
                variable = tk.IntVar(value=default)
                widget = tk.Scale(self.parameters, variable=variable, from_=minimum, to=maximum, resolution=step, orient="horizontal", length=220, showvalue=True, bg="#2b3138", fg="#f1f3f5", highlightthickness=0, troughcolor="#111315", command=lambda _value: self._refresh_preview())
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.parameters.columnconfigure(1, weight=1)
            self.parameter_vars[name] = variable
            self.parameter_widgets.append(widget)

    def _value(self, name: str) -> int | str:
        return self.parameter_vars[name].get()

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        name = self.detection_var.get()
        if name == "Original":
            return frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if name == "Canny edge detection":
            return cv2.Canny(gray, int(self._value("low")), int(self._value("high")))
        if name == "Sobel edge detection":
            kernel = int(self._value("kernel"))
            horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
            vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
            direction = self._value("direction")
            result = horizontal if direction == "Horizontal" else vertical if direction == "Vertical" else cv2.magnitude(horizontal, vertical)
            return cv2.convertScaleAbs(result)
        if name == "Laplacian edges":
            result = cv2.Laplacian(gray, cv2.CV_64F, ksize=int(self._value("kernel")))
            result = cv2.convertScaleAbs(result)
            return cv2.threshold(result, int(self._value("threshold")), 255, cv2.THRESH_BINARY)[1]
        if name == "Hough circle detection":
            circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, int(self._value("dp")) / 10, int(self._value("min_distance")), param1=int(self._value("param1")), param2=int(self._value("param2")), minRadius=int(self._value("min_radius")), maxRadius=int(self._value("max_radius")))
            result = frame.copy()
            if circles is not None:
                for x, y, radius in np.round(circles[0]).astype(int):
                    cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
                    cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            return result
        if name == "Hough line detection":
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(self._value("threshold")), minLineLength=int(self._value("min_length")), maxLineGap=int(self._value("max_gap")))
            result = frame.copy()
            if lines is not None:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return result
        if name == "Contour detection":
            mask = cv2.threshold(gray, int(self._value("threshold")), 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return cv2.drawContours(frame.copy(), [c for c in contours if cv2.contourArea(c) >= int(self._value("min_area"))], -1, (0, 255, 0), 2)
        if name in ("Corner detection", "Shi-Tomasi tracking points"):
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=int(self._value("max_corners")), qualityLevel=int(self._value("quality")) / 100, minDistance=int(self._value("min_distance")))
            result = frame.copy()
            if corners is not None:
                for x, y in np.int32(corners).reshape(-1, 2):
                    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
            return result
        if name == "Blob detection":
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = float(self._value("min_area"))
            params.maxArea = float(self._value("max_area"))
            params.filterByCircularity = True
            params.minCircularity = int(self._value("circularity")) / 100
            detector = cv2.SimpleBlobDetector_create(params)
            return cv2.drawKeypoints(frame, detector.detect(gray), None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if name == "Watershed segmentation":
            binary = cv2.threshold(gray, int(self._value("threshold")), 255, cv2.THRESH_BINARY_INV)[1]
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            markers = cv2.connectedComponents(np.uint8(distance > 0.4 * distance.max()))[1] + 1
            markers[binary == 0] = 0
            result = frame.copy()
            markers = cv2.watershed(result, markers)
            result[markers == -1] = (0, 0, 255)
            return result
        if name == "Foreground mask (MOG2)":
            if self.mog2 is None:
                self.mog2 = cv2.createBackgroundSubtractorMOG2(history=int(self._value("history")), varThreshold=int(self._value("variance")), detectShadows=True)
            return self.mog2.apply(frame)
        return frame.copy()

    def _refresh_preview(self) -> None:
        if self.current_frame is None:
            return
        try:
            result = self._detect(self.current_frame)
        except cv2.error as error:
            self.status_var.set(f"Detection error: {error.code}")
            return
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        width, height = max(400, self.preview.winfo_width()), max(300, self.preview.winfo_height())
        scale = min(width / result.shape[1], height / result.shape[0], 1)
        result = cv2.resize(result, (max(1, int(result.shape[1] * scale)), max(1, int(result.shape[0] * scale))))
        success, encoded = cv2.imencode(".ppm", result)
        if success:
            self.preview_photo = tk.PhotoImage(data=encoded.tobytes(), format="PPM")
            self.preview.configure(image=self.preview_photo, text="")

    def _save_result(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("Save", "Select an image or video first.")
            return
        result = self._detect(self.current_frame)
        target = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if target and not cv2.imwrite(target, result):
            messagebox.showerror("Save", f"Could not save:\n{target}")
        elif target:
            self.status_var.set(f"Saved {Path(target).name}")


def main() -> None:
    root = tk.Tk()
    app = DetectionWorkbench(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_video(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
