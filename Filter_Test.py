"""Interactive OpenCV image and video filter workbench."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class OpenCVWorkbench:
    """Small desktop UI for exploring OpenCV filters on local media."""

    FILTERS = {
        "Original": {},
        "Grayscale": {},
        "Gaussian blur": {"kernel": (1, 31, 2, 5)},
        "Median blur": {"kernel": (1, 31, 2, 5)},
        "Bilateral filter": {"diameter": (1, 20, 1, 7), "sigma_color": (1, 200, 1, 75), "sigma_space": (1, 200, 1, 75)},
        "Sharpen": {"amount": (0, 30, 1, 10)},
        "Canny edges": {"low": (0, 255, 1, 50), "high": (1, 255, 1, 150)},
        "Threshold": {"threshold": (0, 255, 1, 127)},
        "Adaptive threshold": {"block_size": (3, 51, 2, 11), "constant": (-20, 20, 1, 2)},
        "Morphological operation": {"operation": ("Erode", "Dilate", "Open", "Close", "Gradient"), "kernel": (1, 31, 2, 3), "iterations": (1, 10, 1, 1)},
        "Emboss": {"strength": (0, 10, 1, 2)},
        "Sepia": {"intensity": (0, 100, 1, 100)},
        "HSV color": {"hue": (0, 179, 1, 0), "saturation": (0, 200, 1, 100), "value": (0, 200, 1, 100)},
        "Brightness / contrast": {"brightness": (-100, 100, 1, 0), "contrast": (0, 300, 1, 100)},
        "Invert": {},
        "Cartoon": {"edges": (1, 20, 1, 9), "color": (1, 15, 1, 7)},
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenCV Photo & Video Filter Workbench")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)

        self.project_dir = Path(__file__).resolve().parent
        self.current_path: Path | None = None
        self.current_frame: np.ndarray | None = None
        self.video: cv2.VideoCapture | None = None
        self.video_is_playing = False
        self.video_fps = 25.0
        self.preview_photo: tk.PhotoImage | None = None
        self.parameter_vars: dict[str, tk.Variable] = {}
        self.parameter_widgets: list[tk.Widget] = []

        self.source_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="Original")
        self.status_var = tk.StringVar(value="Choose an image or video to begin.")
        self.play_var = tk.StringVar(value="Play")

        self._configure_style()
        self._build_ui()
        self._load_media()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#20252b")
        style.configure("Panel.TFrame", background="#2b3138")
        style.configure("TLabel", background="#2b3138", foreground="#f1f3f5")
        style.configure("Title.TLabel", background="#20252b", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", background="#20252b", foreground="#aab2bc")
        style.configure("TButton", padding=(9, 5))
        style.configure("TCombobox", padding=4)

    def _build_ui(self) -> None:
        ttk.Label(self.root, text="OpenCV Photo & Video Filter Workbench", style="Title.TLabel").pack(
            anchor="w", padx=18, pady=(14, 2)
        )
        ttk.Label(
            self.root,
            text="Select media, experiment with OpenCV parameters, and save the processed preview.",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sources = ttk.Frame(main, style="Panel.TFrame", padding=12)
        sources.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(sources, text="Media sources", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(sources, text="data images and Broodjes_MP4 videos", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        self.source_list = tk.Listbox(
            sources, width=30, height=20, bg="#171a1e", fg="#e8eaed", selectbackground="#3b82f6",
            borderwidth=0, highlightthickness=0, activestyle="none",
        )
        self.source_list.pack(fill="both", expand=True)
        self.source_list.bind("<<ListboxSelect>>", self._select_media)
        ttk.Button(sources, text="Refresh sources", command=self._load_media).pack(fill="x", pady=(10, 4))
        ttk.Button(sources, text="Open file...", command=self._open_file).pack(fill="x")

        work = ttk.Frame(main, style="Panel.TFrame", padding=12)
        work.grid(row=0, column=1, sticky="nsew")
        work.columnconfigure(0, weight=1)
        work.rowconfigure(1, weight=1)

        top = ttk.Frame(work, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(top, text="Filter:").pack(side="left")
        filter_box = ttk.Combobox(top, textvariable=self.filter_var, values=list(self.FILTERS), state="readonly", width=28)
        filter_box.pack(side="left", padx=(8, 14))
        filter_box.bind("<<ComboboxSelected>>", self._filter_changed)
        ttk.Button(top, textvariable=self.play_var, command=self._toggle_video).pack(side="left")
        ttk.Button(top, text="Save processed...", command=self._save_processed).pack(side="right")

        self.preview = tk.Label(work, text="No media selected", bg="#111315", fg="#8d98a5")
        self.preview.grid(row=1, column=0, sticky="nsew")

        bottom = ttk.Frame(work, style="Panel.TFrame")
        bottom.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        self.parameters = ttk.Frame(bottom, style="Panel.TFrame")
        self.parameters.grid(row=0, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.status_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._rebuild_parameters()

    def _load_media(self) -> None:
        self.source_list.delete(0, tk.END)
        self.media_paths: list[Path] = []
        for folder, extensions in (
            (self.project_dir / "data", IMAGE_EXTENSIONS),
            (self.project_dir / "Broodjes_MP4", VIDEO_EXTENSIONS),
        ):
            if folder.exists():
                self.media_paths.extend(sorted(path for path in folder.rglob("*") if path.suffix.lower() in extensions))
        for path in self.media_paths:
            relative = path.relative_to(self.project_dir)
            self.source_list.insert(tk.END, str(relative))
        self.status_var.set(f"{len(self.media_paths)} media files found.")

    def _open_file(self) -> None:
        selected = filedialog.askopenfilename(
            filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        )
        if selected:
            self._set_media(Path(selected))

    def _select_media(self, _event: object = None) -> None:
        selection = self.source_list.curselection()
        if selection:
            self._set_media(self.media_paths[selection[0]])

    def _set_media(self, path: Path) -> None:
        self._stop_video()
        self.current_path = path
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            self.video = cv2.VideoCapture(str(path))
            if not self.video.isOpened():
                messagebox.showerror("OpenCV", f"Could not open video:\n{path}")
                self.video = None
                return
            self.video_fps = self.video.get(cv2.CAP_PROP_FPS) or 25.0
            self.play_var.set("Play")
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
            if self.video_is_playing:
                delay = max(1, round(1000 / self.video_fps))
                self.root.after(delay, self._read_video_frame)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_is_playing = False
            self.play_var.set("Play")

    def _toggle_video(self) -> None:
        if self.video is None:
            return
        self.video_is_playing = not self.video_is_playing
        self.play_var.set("Pause" if self.video_is_playing else "Play")
        if self.video_is_playing:
            self._read_video_frame()

    def _stop_video(self) -> None:
        self.video_is_playing = False
        if self.video is not None:
            self.video.release()
            self.video = None
        self.play_var.set("Play")

    def _filter_changed(self, _event: object = None) -> None:
        self._rebuild_parameters()
        self._refresh_preview()

    def _rebuild_parameters(self) -> None:
        for widget in self.parameter_widgets:
            widget.destroy()
        self.parameter_widgets.clear()
        self.parameter_vars.clear()
        definitions = self.FILTERS[self.filter_var.get()]
        for index, (name, definition) in enumerate(definitions.items()):
            label = name.replace("_", " ").title()
            ttk.Label(self.parameters, text=f"{label}:").grid(row=index, column=0, sticky="w", padx=(0, 8), pady=2)
            if isinstance(definition[0], str):
                variable = tk.StringVar(value=definition[0])
                widget = ttk.Combobox(self.parameters, textvariable=variable, values=definition, state="readonly", width=14)
                widget.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preview())
            else:
                minimum, maximum, step, default = definition
                variable = tk.IntVar(value=default)
                widget = tk.Scale(
                    self.parameters, variable=variable, from_=minimum, to=maximum, resolution=step,
                    orient="horizontal", length=220, showvalue=True, bg="#2b3138", fg="#f1f3f5",
                    highlightthickness=0, troughcolor="#111315", command=lambda _value: self._refresh_preview(),
                )
            widget.grid(row=index, column=1, sticky="ew", pady=2)
            self.parameters.columnconfigure(1, weight=1)
            self.parameter_vars[name] = variable
            self.parameter_widgets.extend((widget,))

    def _value(self, name: str) -> int | str:
        return self.parameter_vars[name].get()

    def _apply_filter(self, frame: np.ndarray) -> np.ndarray:
        name = self.filter_var.get()
        if name == "Original":
            return frame.copy()
        if name == "Grayscale":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if name == "Gaussian blur":
            return cv2.GaussianBlur(frame, (int(self._value("kernel")),) * 2, 0)
        if name == "Median blur":
            return cv2.medianBlur(frame, int(self._value("kernel")))
        if name == "Bilateral filter":
            return cv2.bilateralFilter(frame, int(self._value("diameter")), int(self._value("sigma_color")), int(self._value("sigma_space")))
        if name == "Sharpen":
            amount = int(self._value("amount")) / 10
            return cv2.addWeighted(frame, 1 + amount, cv2.GaussianBlur(frame, (0, 0), 3), -amount, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if name == "Canny edges":
            return cv2.Canny(gray, int(self._value("low")), int(self._value("high")))
        if name == "Threshold":
            return cv2.threshold(gray, int(self._value("threshold")), 255, cv2.THRESH_BINARY)[1]
        if name == "Adaptive threshold":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(self._value("block_size")), int(self._value("constant")))
        if name == "Morphological operation":
            kernel = np.ones((int(self._value("kernel")),) * 2, np.uint8)
            operation = self._value("operation")
            operations = {"Erode": cv2.erode, "Dilate": cv2.dilate, "Open": cv2.MORPH_OPEN, "Close": cv2.MORPH_CLOSE, "Gradient": cv2.MORPH_GRADIENT}
            if operation in ("Erode", "Dilate"):
                return operations[operation](frame, kernel, iterations=int(self._value("iterations")))
            return cv2.morphologyEx(frame, operations[operation], kernel, iterations=int(self._value("iterations")))
        if name == "Emboss":
            strength = int(self._value("strength"))
            kernel = np.array([[-strength, -1, 0], [-1, 1, 1], [0, 1, strength]], dtype=np.float32)
            return cv2.filter2D(frame, -1, kernel) + 128
        if name == "Sepia":
            sepia = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
            result = cv2.transform(frame, sepia)
            intensity = int(self._value("intensity")) / 100
            return cv2.addWeighted(frame, 1 - intensity, np.clip(result, 0, 255).astype(np.uint8), intensity, 0)
        if name == "HSV color":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + int(self._value("hue"))) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * int(self._value("saturation")) / 100, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * int(self._value("value")) / 100, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if name == "Brightness / contrast":
            return cv2.convertScaleAbs(frame, alpha=int(self._value("contrast")) / 100, beta=int(self._value("brightness")))
        if name == "Invert":
            return cv2.bitwise_not(frame)
        if name == "Cartoon":
            color_sigma = int(self._value("color")) * 15
            colors = cv2.bilateralFilter(frame, 9, color_sigma, color_sigma)
            edges = cv2.adaptiveThreshold(cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2 * int(self._value("edges")) + 1, 9)
            return cv2.bitwise_and(colors, colors, mask=edges)
        return frame.copy()

    def _refresh_preview(self) -> None:
        if self.current_frame is None:
            return
        result = self._apply_filter(self.current_frame)
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        max_width, max_height = max(400, self.preview.winfo_width()), max(300, self.preview.winfo_height())
        scale = min(max_width / result.shape[1], max_height / result.shape[0], 1)
        resized = cv2.resize(result, (max(1, int(result.shape[1] * scale)), max(1, int(result.shape[0] * scale))))
        success, encoded = cv2.imencode(".ppm", resized)
        if success:
            self.preview_photo = tk.PhotoImage(data=encoded.tobytes(), format="PPM")
            self.preview.configure(image=self.preview_photo, text="")

    def _save_processed(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("Save", "Select an image or video first.")
            return
        result = self._apply_filter(self.current_frame)
        extension = ".png" if self.current_path is None or self.current_path.suffix.lower() not in VIDEO_EXTENSIONS else ".jpg"
        target = filedialog.asksaveasfilename(defaultextension=extension, filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if target and not cv2.imwrite(target, result):
            messagebox.showerror("Save", f"Could not save:\n{target}")
        elif target:
            self.status_var.set(f"Saved {Path(target).name}")


def main() -> None:
    root = tk.Tk()
    app = OpenCVWorkbench(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_video(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
