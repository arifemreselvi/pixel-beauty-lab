from __future__ import annotations

import sys
import threading
import traceback
import ctypes
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


def resolve_model_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "models"
        if bundled.exists():
            return bundled

    here = Path(__file__).resolve().parent
    local_models = here / "models"
    if local_models.exists():
        return local_models

    exe_models = Path(sys.executable).resolve().parent / "models"
    if exe_models.exists():
        return exe_models

    return local_models


class FusionDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fusion Upscaler")
        self.root.geometry("1200x760")
        self.root.minsize(980, 680)

        self.bg = "#0b0f14"
        self.panel = "#111722"
        self.card = "#171f2c"
        self.border = "#2b3748"
        self.accent = "#c29a62"
        self.text = "#e8edf6"
        self.muted = "#9dabc0"

        self.root.configure(bg=self.bg)
        self._enable_dark_titlebar()

        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None
        self._input_preview_ref: ImageTk.PhotoImage | None = None
        self._output_preview_ref: ImageTk.PhotoImage | None = None
        self.upscaler: FusionUpscaler | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.bg)
        style.configure("Card.TFrame", background=self.card)
        style.configure("Panel.TFrame", background=self.panel)
        style.configure("TLabel", background=self.panel, foreground=self.text)
        style.configure("Muted.TLabel", background=self.panel, foreground=self.muted)
        style.configure("TCombobox", fieldbackground="#0f1521", background="#0f1521", foreground=self.text, bordercolor=self.border, lightcolor=self.border, darkcolor=self.border, arrowcolor=self.accent)
        style.map("TCombobox", fieldbackground=[("readonly", "#0f1521")], foreground=[("readonly", self.text)], selectbackground=[("readonly", "#0f1521")], selectforeground=[("readonly", self.text)])
        style.configure("Horizontal.TScale", background=self.card, troughcolor="#273247")
        style.configure("TButton", font=("Segoe UI", 10))

        shell = ttk.Frame(self.root, style="Panel.TFrame", padding=18)
        shell.pack(fill="both", expand=True)

        title = ttk.Label(shell, text="Fusion Upscaler", font=("Cambria", 25, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            shell,
            text="Desktop 4x upscaling with smart model routing and premium detail enhancement.",
            style="Muted.TLabel",
            font=("Segoe UI", 11),
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        top_line = tk.Frame(shell, bg=self.accent, height=2)
        top_line.pack(fill="x", pady=(0, 14))

        content = ttk.Frame(shell, style="Panel.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="Card.TFrame", padding=12)
        right = ttk.Frame(content, style="Card.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.input_canvas = tk.Canvas(left, bg="#101521", highlightthickness=1, highlightbackground=self.border)
        self.input_canvas.pack(fill="both", expand=True)
        self._draw_placeholder(self.input_canvas, "Upload image")

        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x", pady=(12, 0))

        ttk.Label(controls, text="Upscale strategy", font=("Segoe UI", 10, "bold"), background=self.card).pack(anchor="w")
        self.mode_var = tk.StringVar(value="Auto")
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["Auto", "Fusion", "Photo", "Anime"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        mode_combo.pack(fill="x", pady=(4, 10))

        self.sharpness_var = tk.DoubleVar(value=0.35)
        ttk.Label(controls, text="Sharpness", background=self.card).pack(anchor="w")
        ttk.Scale(controls, from_=0.0, to=1.0, variable=self.sharpness_var).pack(fill="x", pady=(2, 10))

        self.quality_var = tk.DoubleVar(value=0.35)
        ttk.Label(controls, text="Quality boost", background=self.card).pack(anchor="w")
        ttk.Scale(controls, from_=0.0, to=1.0, variable=self.quality_var).pack(fill="x", pady=(2, 12))

        btn_row = ttk.Frame(controls, style="Card.TFrame")
        btn_row.pack(fill="x")
        self.open_btn = tk.Button(btn_row, text="Open Image", command=self.open_image, bg="#8f6b3f", fg="white", relief="flat")
        self.open_btn.pack(side="left", padx=(0, 8))
        self.run_btn = tk.Button(btn_row, text="Upscale x4", command=self.run_upscale, bg=self.accent, fg="white", relief="flat")
        self.run_btn.pack(side="left", padx=(0, 8))
        self.save_btn = tk.Button(btn_row, text="Save Output", command=self.save_output, bg="#5f6a73", fg="white", relief="flat")
        self.save_btn.pack(side="left")

        self.output_canvas = tk.Canvas(right, bg="#101521", highlightthickness=1, highlightbackground=self.border)
        self.output_canvas.pack(fill="both", expand=True)
        self._draw_placeholder(self.output_canvas, "Upscaled output")

        ttk.Label(right, text="Process details", font=("Segoe UI", 10, "bold"), background=self.card).pack(anchor="w", pady=(10, 4))
        self.details = tk.Text(right, height=9, wrap="word", bg="#111724", fg=self.text, bd=1, relief="solid", insertbackground=self.text)
        self.details.pack(fill="x")
        self.details.insert("1.0", "Ready.")
        self.details.config(state="disabled")

        self.status_var = tk.StringVar(value="Idle")
        status = ttk.Label(shell, textvariable=self.status_var, style="Muted.TLabel", font=("Segoe UI", 10))
        status.pack(anchor="w", pady=(10, 0))

    def _enable_dark_titlebar(self) -> None:
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            dwm = ctypes.windll.dwmapi
            res = dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
            if res != 0:
                dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

    def _draw_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        canvas.create_text(
            canvas.winfo_width() // 2 if canvas.winfo_width() > 1 else 250,
            canvas.winfo_height() // 2 if canvas.winfo_height() > 1 else 150,
            text=text,
            fill="#b8bac1",
            font=("Segoe UI", 16, "bold"),
        )

    def _set_details(self, text: str) -> None:
        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.config(state="disabled")

    @staticmethod
    def _fit_preview(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = image.size
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    def _render_image(self, canvas: tk.Canvas, image: Image.Image, is_output: bool) -> None:
        canvas.update_idletasks()
        max_w = max(canvas.winfo_width() - 10, 120)
        max_h = max(canvas.winfo_height() - 10, 120)
        preview = self._fit_preview(image, max_w, max_h)
        photo = ImageTk.PhotoImage(preview)

        canvas.delete("all")
        canvas.create_image(max_w // 2 + 5, max_h // 2 + 5, image=photo)
        if is_output:
            self._output_preview_ref = photo
        else:
            self._input_preview_ref = photo

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        try:
            self.input_image = Image.open(path).convert("RGB")
            self._render_image(self.input_canvas, self.input_image, is_output=False)
            self.status_var.set(f"Loaded: {Path(path).name}")
            self._set_details("Image loaded. Press 'Upscale x4' to process.")
        except Exception as exc:
            messagebox.showerror("Fusion Upscaler", f"Could not open image.\n\n{exc}")

    def _load_upscaler_if_needed(self) -> None:
        if self.upscaler is None:
            from fusion_upscaler import FusionUpscaler

            model_dir = resolve_model_dir()
            self.upscaler = FusionUpscaler(model_dir)

    def run_upscale(self) -> None:
        if self.input_image is None:
            messagebox.showwarning("Fusion Upscaler", "Please open an image first.")
            return

        self.run_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.status_var.set("Upscaling... please wait")
        self._set_details("Processing...")

        worker = threading.Thread(target=self._upscale_worker, daemon=True)
        worker.start()

    def _upscale_worker(self) -> None:
        try:
            self._load_upscaler_if_needed()
            assert self.upscaler is not None
            arr = np.array(self.input_image, dtype=np.uint8)
            result = self.upscaler.upscale(
                arr,
                mode=self.mode_var.get(),
                sharpness=float(self.sharpness_var.get()),
                quality=float(self.quality_var.get()),
            )
            out_image = Image.fromarray(result.image)
            details = (
                f"Route: {result.route_note}\n"
                f"Mode: {result.model_used}\n"
                f"Time: {result.elapsed_seconds:.2f}s\n"
                f"Output size: {out_image.width} x {out_image.height}"
            )
            self.root.after(0, self._on_upscale_success, out_image, details)
        except Exception as exc:
            trace = traceback.format_exc(limit=6)
            self.root.after(0, self._on_upscale_error, exc, trace)

    def _on_upscale_success(self, out_image: Image.Image, details: str) -> None:
        self.output_image = out_image
        self._render_image(self.output_canvas, self.output_image, is_output=True)
        self._set_details(details)
        self.status_var.set("Done")
        self.run_btn.config(state="normal")
        self.open_btn.config(state="normal")
        self.save_btn.config(state="normal")

    def _on_upscale_error(self, exc: Exception, trace: str) -> None:
        self.run_btn.config(state="normal")
        self.open_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.status_var.set("Error")
        self._set_details(f"Error:\n{exc}\n\n{trace}")
        messagebox.showerror("Fusion Upscaler", f"Upscaling failed.\n\n{exc}")

    def save_output(self) -> None:
        if self.output_image is None:
            messagebox.showinfo("Fusion Upscaler", "No output yet. Upscale an image first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save upscaled image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("WebP", "*.webp")],
        )
        if not path:
            return
        self.output_image.save(path)
        self.status_var.set(f"Saved: {Path(path).name}")


def main() -> None:
    root = tk.Tk()
    app = FusionDesktopApp(root)
    root.after(200, lambda: app._draw_placeholder(app.input_canvas, "Upload image"))
    root.after(220, lambda: app._draw_placeholder(app.output_canvas, "Upscaled output"))
    root.mainloop()


if __name__ == "__main__":
    main()
