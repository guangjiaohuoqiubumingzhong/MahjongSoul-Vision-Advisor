"""Desktop launcher; run with this project's .venv Python."""
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
try:
    from app.ui.advice import launch
    launch(ROOT / "configs/window-steam-1366-v2.json",ROOT / "reports/screenshot-advice")
except Exception as exc:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("无法启动截图建议",str(exc),parent=root)
    root.destroy()
