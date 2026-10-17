"""Entry point for the self-contained Windows release."""
import os
from pathlib import Path
import sys
import traceback


def main():
    frozen = getattr(sys, "frozen", False)
    resources = Path(sys._MEIPASS) if frozen else Path(__file__).resolve().parent
    embedded = (resources / "portable-runtime.json").is_file()
    folder = Path(sys.executable).resolve().parent if frozen else (resources.parent if embedded else resources)
    userdata = folder / "userdata"
    userdata.mkdir(parents=True, exist_ok=True)
    os.chdir(resources)
    if embedded:
        os.environ["TCL_LIBRARY"] = str(resources / "_tcl_data")
        os.environ["TK_LIBRARY"] = str(resources / "_tk_data")
    # Windowed executables have no console streams. Keep native-library messages
    # and uncaught startup errors available when testing on another computer.
    log = (userdata / "startup.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    try:
        from app.portable import run
        return run(resources, userdata)
    except Exception as exc:
        traceback.print_exc(file=log)
        if "--self-test" not in sys.argv:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Majan 无法启动", f"{exc}\n\n详细记录：{userdata / 'startup.log'}", parent=root)
            root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
