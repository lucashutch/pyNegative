#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import json
from PIL import Image, ImageTk
import pyrawroom

# --- Core Processing Logic ---


class RawEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAW Tone Editor")
        self.root.geometry("1300x900")

        self.raw_path = None
        self.base_img_full = None
        self.base_img_preview = None
        self.preview_image_tk = None

        # --- Layout ---
        self.panel = ttk.Frame(root, padding=10, width=350)
        self.panel.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2b2b2b")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Controls ---
        # File Operations Frame
        file_frame = ttk.LabelFrame(self.panel, text="File Operations", padding=5)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="Open RAW File", command=self.browse_raw_file).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            file_frame, text="Load Edit (.json)", command=self.load_json_edit
        ).pack(fill=tk.X, pady=2)

        self.lbl_info = ttk.Label(self.panel, text="No file loaded", wraplength=280)
        self.lbl_info.pack(pady=5)

        ttk.Separator(self.panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # Basic Tone Section
        self.add_slider("Exposure", -4.0, 4.0, 0.0, "val_exposure")
        self.add_slider("Contrast (Whites)", 0.5, 2.0, 1.0, "val_whites")
        self.add_slider("Blacks", -0.2, 0.2, 0.0, "val_blacks")

        ttk.Separator(self.panel, orient="horizontal").pack(fill=tk.X, pady=10)

        self.add_slider("Highlights", -1.0, 1.0, 0.0, "val_highlights")
        self.add_slider("Shadows", -1.0, 1.0, 0.0, "val_shadows")

        ttk.Separator(self.panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # Sharpening
        self.var_sharpen = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.panel,
            text="Enable Sharpening",
            variable=self.var_sharpen,
            command=self.request_update,
        ).pack(anchor="w")
        self.add_slider("Sharpen Radius", 0.5, 5.0, 2.0, "val_radius")
        self.add_slider("Sharpen Amount", 0, 300, 150, "val_percent", is_int=True)

        ttk.Separator(self.panel, orient="horizontal").pack(fill=tk.X, pady=20)

        self.btn_save = ttk.Button(
            self.panel, text="Save Result", command=self.save_file, state="disabled"
        )
        self.btn_save.pack(fill=tk.X, pady=10)

    def add_slider(self, label, min_val, max_val, default, var_name, is_int=False):
        frame = ttk.Frame(self.panel)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=label).pack(anchor="w")

        if is_int:
            var = tk.IntVar(value=default)
        else:
            var = tk.DoubleVar(value=default)

        setattr(self, var_name, var)

        scale = ttk.Scale(
            frame, from_=min_val, to=max_val, variable=var, command=self.request_update
        )
        scale.pack(fill=tk.X)

        lbl = ttk.Label(frame, text=str(default))
        lbl.pack(anchor="e")
        setattr(self, f"lbl_{var_name}", lbl)

    # --- Loading Logic ---

    def browse_raw_file(self):
        """User browses for a RAW file manually."""
        path = filedialog.askopenfilename(
            filetypes=[("RAW files", " ".join(f"*{ext}" for ext in pyrawroom.SUPPORTED_EXTS))]
        )
        if not path:
            return
        self.load_raw_image(path)

    def load_raw_image(self, path):
        """Loads the actual RAW data and creates previews."""
        self.lbl_info.config(text=f"Loading: {os.path.basename(path)}...")
        self.root.update()

        try:
            self.base_img_full = pyrawroom.open_raw(path)
            self.raw_path = path

            # Preview resizing
            h, w, _ = self.base_img_full.shape
            scale = 1000 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            temp_pil = Image.fromarray((self.base_img_full * 255).astype(np.uint8))
            temp_pil = temp_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self.base_img_preview = np.array(temp_pil).astype(np.float32) / 255.0

            self.lbl_info.config(text=f"Loaded: {os.path.basename(path)}")
            self.btn_save.config(state="normal")
            self.request_update()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            self.lbl_info.config(text="Error loading file")


    def load_json_edit(self):
        """Loads settings from a JSON file and tries to find the associated RAW file."""
        json_path = filedialog.askopenfilename(filetypes=[("JSON Edit", "*.json")])
        if not json_path:
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # 1. Find the RAW file
            original_path = data.get("original_file", "")

            # Strategy A: Check exact path
            final_raw_path = original_path

            if not os.path.exists(final_raw_path):
                # Strategy B: Check if it's in the same folder as the JSON file
                # (Useful if you moved the folder containing both jpg/json/raw)
                json_dir = os.path.dirname(json_path)
                raw_filename = os.path.basename(original_path)
                potential_path = os.path.join(json_dir, raw_filename)

                if os.path.exists(potential_path):
                    final_raw_path = potential_path
                else:
                    messagebox.showerror(
                        "Error", f"Could not find original RAW file:\n{original_path}"
                    )
                    return

            # 2. Load the image
            self.load_raw_image(final_raw_path)

            # 3. Apply Settings
            settings = data.get("settings", {})
            self.val_exposure.set(settings.get("exposure", 0.0))
            self.val_whites.set(settings.get("whites", 1.0))
            self.val_blacks.set(settings.get("blacks", 0.0))
            self.val_highlights.set(settings.get("highlights", 0.0))
            self.val_shadows.set(settings.get("shadows", 0.0))

            self.var_sharpen.set(settings.get("sharpen_enabled", False))
            self.val_radius.set(settings.get("sharpen_radius", 2.0))
            self.val_percent.set(settings.get("sharpen_percent", 150))

            # Refresh
            self.request_update()
            self.lbl_info.config(text=f"Loaded Edit: {os.path.basename(json_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON edit:\n{e}")

    # --- Processing & Saving ---

    def request_update(self, event=None):
        if self.base_img_preview is None:
            return

        for name in [
            "val_exposure",
            "val_whites",
            "val_blacks",
            "val_highlights",
            "val_shadows",
            "val_radius",
        ]:
            lbl = getattr(self, f"lbl_{name}")
            val = getattr(self, name).get()
            lbl.config(text=f"{val:.2f}")

        self.lbl_val_percent.config(text=str(getattr(self, "val_percent").get()))
        self.update_preview()

    def process_image(self, img_arr):
        img, _ = pyrawroom.apply_tone_map(
            img_arr,
            exposure=self.val_exposure.get(),
            blacks=self.val_blacks.get(),
            whites=self.val_whites.get(),
            shadows=self.val_shadows.get(),
            highlights=self.val_highlights.get(),
        )

        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        if self.var_sharpen.get():
            pil_img = pyrawroom.sharpen_image(
                pil_img,
                radius=self.val_radius.get(),
                percent=self.val_percent.get()
            )
        return pil_img

    def update_preview(self):
        res_img = self.process_image(self.base_img_preview)
        self.preview_image_tk = ImageTk.PhotoImage(res_img)

        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        self.canvas.delete("all")
        self.canvas.create_image(
            c_w // 2, c_h // 2, anchor=tk.CENTER, image=self.preview_image_tk
        )

    def save_file(self):
        if self.base_img_full is None:
            return

        # Default filename logic
        input_dir = os.path.dirname(self.raw_path)
        base_name = os.path.splitext(os.path.basename(self.raw_path))[0]
        default_filename = f"{base_name}.jpg"

        out_path = filedialog.asksaveasfilename(
            initialdir=input_dir,
            initialfile=default_filename,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("HEIF", "*.heic")],
        )
        if not out_path:
            return

        self.lbl_info.config(text="Processing full res... please wait.")
        self.root.update()

        try:
            # 1. Save Image
            final_img = self.process_image(self.base_img_full)
            pyrawroom.save_image(final_img, out_path, quality=95)

            # 2. Save JSON Sidecar
            json_path = os.path.splitext(out_path)[0] + ".json"
            settings_data = {
                "original_file": self.raw_path,
                "settings": {
                    "exposure": self.val_exposure.get(),
                    "whites": self.val_whites.get(),
                    "blacks": self.val_blacks.get(),
                    "highlights": self.val_highlights.get(),
                    "shadows": self.val_shadows.get(),
                    "sharpen_enabled": self.var_sharpen.get(),
                    "sharpen_radius": self.val_radius.get(),
                    "sharpen_percent": self.val_percent.get(),
                },
            }
            with open(json_path, "w") as f:
                json.dump(settings_data, f, indent=4)

            messagebox.showinfo(
                "Saved", f"Saved Image & Settings to:\n{os.path.dirname(out_path)}"
            )
            self.lbl_info.config(text=f"Saved: {os.path.basename(out_path)}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = RawEditorApp(root)
    root.mainloop()
