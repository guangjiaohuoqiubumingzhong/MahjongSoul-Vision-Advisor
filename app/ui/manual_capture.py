"""Human-operated screenshot collection. Never sends input to the game."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk,messagebox
import uuid

import psutil

from app.capture.frames import WindowCapture
from app.capture.window import CaptureError,ClientArea,windows
from app.vision.config import load_config,write_json

WORKSPACE=Path(__file__).resolve().parents[2]
DEFAULT_CLIENT=Path(r"F:\Steam\steamapps\common\MahjongSoul\Jantama_MahjongSoul.exe")


@dataclass(frozen=True)
class CaptureRequest:
    hwnd:int
    pid:int
    game_id:str
    split:str
    output:Path
    client_exe:Path


def process_exe(pid):
    return Path(psutil.Process(pid).exe())


def game_windows(client_exe):
    found=[]
    failures=[]
    for area in windows("雀魂"):
        try:
            if str(process_exe(area.pid)).casefold()==str(client_exe).casefold():
                found.append(area)
        except (psutil.Error,OSError) as exc:
            failures.append(type(exc).__name__)
            continue
    if not found and failures:
        raise CaptureError("找到了雀魂窗口，但无权限核对进程路径；请从普通用户终端启动采集窗口（不要在隔离环境中运行）。")
    return found


def validate_split(game_id,split):
    if not game_id.strip() or split not in {"train","debug","test"}:
        raise ValueError("请填写对局编号并选择 train/debug/test。")
    for path in (WORKSPACE/"data").rglob("*.annotation.json"):
        annotation=json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("game_id")==game_id and annotation.get("split")!=split:
            raise ValueError("同一对局已有不同的数据划分；不能将验收牌谱改为训练素材。")


def capture_one(request,config):
    """Called only after the human's button click, in the MSS owning thread."""
    area=next((a for a in game_windows(request.client_exe) if a.hwnd==request.hwnd),None)
    if area is None or area.pid!=request.pid:
        raise CaptureError("雀魂窗口已关闭或更换，请重新选择；未截取其它窗口。")
    validate_split(request.game_id,request.split)
    capture=WindowCapture(request.hwnd,config)
    try:
        if capture.initial.pid!=request.pid:
            raise CaptureError("窗口进程已更换，暂停采集。")
        frame=capture.grab()
        frame.metadata.update(game_id=request.game_id,split=request.split,
                              capture_trigger="human_button_click_after_countdown",
                              game_inputs_sent=0)
        image=frame.save(request.output)
        write_json(image.with_suffix(".annotation.json"),{
            "schema_version":1,"image":str(image.resolve()),"game_id":request.game_id,
            "split":request.split,"reviewed":False,"frame_quality":"unknown",
            "complete_regions":[],"targets":[],"config_version":config["version"],
            "information_scope":"self_hand_and_public_information_only",
            "notes":"Human-triggered capture. Opponent-hand hiding and all labels still require visual review; not an acceptance sample yet."})
        return image
    finally:
        capture.close()


class ManualCapture:
    def __init__(self,root,config_path,output,client_exe,game_id,split):
        self.root=root
        self.config_path=Path(config_path).resolve()
        self.config=load_config(self.config_path)
        self.output=Path(output).resolve()
        if not self.output.is_relative_to(WORKSPACE/"data/screenshots"):
            raise ValueError("人工采集保存目录需位于本项目 data/screenshots 下。")
        self.client_exe=Path(client_exe)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="manual-capture")
        self.future=None
        self.countdown=None
        self.areas=[]
        self.last_image=None
        root.title("Majan · 由你操作的截图采集")
        root.geometry("660x440")
        panel=ttk.Frame(root,padding=18)
        panel.pack(fill="both",expand=True)
        ttk.Label(panel,text="由你切换、播放或暂停雀魂；本工具只在你按按钮后截图。",wraplength=615).pack(anchor="w",pady=5)
        ttk.Label(panel,text="保持客户区 1366×768、默认牌面，关闭其他三家手牌。\n采集按钮会隐藏本窗口 3 秒，请自行切到稳定且无遮挡的游戏画面。",wraplength=615).pack(anchor="w",pady=5)
        self.window=tk.StringVar()
        self.selector=ttk.Combobox(panel,textvariable=self.window,state="readonly")
        self.selector.pack(fill="x",pady=5)
        ttk.Button(panel,text="刷新雀魂窗口",command=self.refresh).pack(anchor="w")
        self.game=tk.StringVar(value=game_id)
        self.split=tk.StringVar(value=split)
        ttk.Label(panel,text="对局编号（该牌谱的全部局和视角保持同一编号）").pack(anchor="w",pady=(10,0))
        ttk.Entry(panel,textvariable=self.game).pack(fill="x")
        ttk.Combobox(panel,textvariable=self.split,values=["train","debug","test"],state="readonly").pack(fill="x",pady=5)
        self.button=ttk.Button(panel,text="3 秒后采集一张",command=self.begin)
        self.button.pack(fill="x",pady=8)
        ttk.Button(panel,text="核对最近截图（打开标注工具）",command=self.review).pack(fill="x")
        self.status=tk.StringVar(value="尚未采集；不会自动操作游戏。")
        ttk.Label(panel,textvariable=self.status,wraplength=615).pack(anchor="w",pady=10)
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.refresh()

    def refresh(self):
        try:
            self.areas=game_windows(self.client_exe)
            self.selector["values"]=[f"{a.title} · {a.width}×{a.height} · hwnd={a.hwnd}" for a in self.areas]
            if len(self.areas)==1:
                self.selector.current(0)
            else:
                self.window.set("")
            self.status.set("请选择当前雀魂窗口。" if self.areas else "没有找到可见雀魂客户区；请你自行打开或还原客户端后刷新。")
        except Exception as exc:
            self.status.set(str(exc))

    def begin(self):
        if self.future is not None or self.countdown is not None:
            return
        try:
            index=self.selector.current()
            if index<0 or index>=len(self.areas):
                raise ValueError("请先选择雀魂窗口。")
            game=self.game.get().strip()
            split=self.split.get()
            validate_split(game,split)
            area=self.areas[index]
            name=datetime.now(timezone.utc).strftime("capture-%Y%m%dT%H%M%SZ-")+uuid.uuid4().hex[:8]
            self.request=CaptureRequest(area.hwnd,area.pid,game,split,self.output/name,self.client_exe)
            self.button.configure(state="disabled")
            self.status.set("3 秒后截图；请你切到雀魂。")
            self.root.withdraw()
            self.countdown=self.root.after(3000,self.submit)
        except Exception as exc:
            messagebox.showerror("无法开始采集",str(exc),parent=self.root)

    def submit(self):
        self.countdown=None
        self.future=self.executor.submit(capture_one,self.request,self.config)
        self.root.after(100,self.poll)

    def poll(self):
        if not self.future.done():
            self.root.after(100,self.poll)
            return
        self.root.deiconify()
        self.button.configure(state="normal")
        try:
            self.last_image=self.future.result()
            self.status.set(f"已保存原图：{self.last_image}\n尚未人工核对，不计入验收。")
            print(json.dumps({"status":"captured","image":str(self.last_image)},ensure_ascii=True),flush=True)
        except Exception as exc:
            self.status.set(f"采集停止：{exc}。请你调整画面后重试。")
            print(json.dumps({"status":"capture_error","reason":str(exc)},ensure_ascii=True),flush=True)
        finally:
            self.future=None

    def review(self):
        if self.last_image is None:
            self.status.set("请先采集一张图。")
            return
        from app.ui.workbench import Workbench
        # Editing this copy cannot overwrite the default or immutable evidence.
        review_config=self.last_image.parent/"review-config.json"
        if not review_config.exists():
            write_json(review_config,self.config)
        Workbench(tk.Toplevel(self.root),self.last_image,review_config)

    def close(self):
        if self.countdown is not None:
            self.root.after_cancel(self.countdown)
            self.countdown=None
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.root.destroy()


def launch(config_path,output,client_exe,game_id,split):
    root=tk.Tk()
    try:
        ManualCapture(root,config_path,output,client_exe,game_id,split)
    except Exception:
        root.destroy()
        raise
    print(json.dumps({"status":"ready","game_id":game_id,"split":split}),flush=True)
    root.mainloop()
