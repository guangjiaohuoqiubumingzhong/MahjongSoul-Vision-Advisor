"""Read-only current-screenshot advice panel. All Tk work stays on the UI thread."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app.advice import ScreenshotAdvice, load_screenshot, save_advice, format_advice, recapture_gaps, recapture_reasons, recapture_until_complete
from app.capture.frames import WindowCapture
from app.capture.window import CaptureError, windows
from app.capture.layout import CLIENT_PRESETS, arrange_client, position_panel
from app.vision.config import load_config, write_json
from app.vision.trigger import TimerEdge
from .tile_text import tile_name, parse_tiles
from .ivory import (AdviceCard, CandidateList, Card, IvoryTheme, PAPER, CARD,
                    INK, MUTED, GREEN, UI, tile)

REGION_NAMES = {"hand":"自己的手牌","draw":"独立摸牌槽","dora":"宝牌指示牌",
                "round":"场次（东1 / 南1）","remaining":"剩余牌数","honba":"本场","kyotaku":"供托"}
SEATS = ["自己","下家","对家","上家"]
KINDS = {"吃":"chi","碰":"pon","明杠":"daiminkan","暗杠":"ankan","加杠":"kakan"}
CAPTURE_DELAY_SECONDS = .3
# A moved, covered or briefly unavailable window is retried at this interval.
CAPTURE_RETRY_SECONDS = .5
ADVICE_RESPONSE_SECONDS = 5.0


class AdviceTiming:
    """Account for waits, all recaptures and queueing, not just the last analysis."""
    def __init__(self):
        self.started=time.perf_counter()
        self.capture_ms=self.analysis_ms=0.0
        self.analysis_count=0

    def grab(self,capture):
        started=time.perf_counter()
        frame=capture.grab()
        self.capture_ms+=(time.perf_counter()-started)*1000
        return frame

    def process(self,service,frame):
        started=time.perf_counter()
        result=service.process(frame.pixels,frame.metadata)
        self.analysis_ms+=(time.perf_counter()-started)*1000
        self.analysis_count+=1
        return result

    def stamp(self,result):
        result['latency']={'trigger_to_result_ms':(time.perf_counter()-self.started)*1000,
                          'capture_ms':self.capture_ms,'analysis_ms':self.analysis_ms,
                          'analysis_count':self.analysis_count,'budget_ms':ADVICE_RESPONSE_SECONDS*1000,
                          'published_monotonic_ns':time.perf_counter_ns()}
        if result.get('recapture_budget_exhausted'):
            result['reason']=(result.get('reason') or '识别尚不完整。')+' 补截已达到时间预算，可点击“刷新截图”重试。'


def candidate_percentages(candidates):
    """Normalize the top five Q scores; displayed tenths add up to 100%."""
    top=sorted(candidates,key=lambda c:-c['model_score'])[:5]
    if not top:return []
    peak=top[0]['model_score']
    weights=[math.exp(candidate['model_score']-peak) for candidate in top]
    total=math.fsum(weights)
    # Keep a 0.1% display floor, then apportion the remaining softmax weight.
    shares=[weight/total*(1000-len(top)) for weight in weights]
    units=[1+math.floor(share) for share in shares]
    remainder=1000-sum(units)
    order=sorted(range(len(top)),key=lambda i:-(shares[i]-math.floor(shares[i])))
    for i in order[:remainder]:units[i]+=1
    return [(candidate,f'{value/10:.1f}%') for candidate,value in zip(top,units)]


def panel_message(text):
    """Do not direct users to review controls removed from this panel."""
    return re.sub('（面板“核对当前局面”[^）]*）','',text)


def action_probe(capture,service,config):
    """Only countdown, hand and own-river regions reach the monitoring loop."""
    timer=capture.grab_region(config['action_timer']['rect'])
    active=service.timer_active(timer.pixels,cropped=True)
    seconds=getattr(service,'timer_seconds',None)
    marker=None
    hand=next((r for r in config.get('regions',{}).values()
               if r['kind']=='hand' and 'hand_layout' in r),None)
    if active and hand and hasattr(service,'action_marker'):
        hand_pixels=capture.grab_region(hand['rect']).pixels
        river=next((r for r in config.get('regions',{}).values()
                    if r['kind']=='river' and 'grid' in r
                    and r.get('player') == config.get('seat_mapping',{}).get('self',0)),None)
        if river:
            marker=service.action_marker(hand_pixels,cropped=True,
                                         river_pixels=capture.grab_region(river['rect']).pixels)
        else:marker=service.action_marker(hand_pixels,cropped=True)
    return active,seconds,marker


def checked_action_probe(capture,service,config,edge,stop):
    """Confirm an action change before publishing or retrying an old frame."""
    reading=action_probe(capture,service,config)
    event=edge.observe(*reading)
    if reading[0] and event is None and (edge.pending_marker is not None or edge.rising):
        if stop.wait(.15) or stop.is_set():
            return (False,None,None),None
        reading=action_probe(capture,service,config)
        event=edge.observe(*reading)
    return reading,event


def region_name(name):
    if name in REGION_NAMES:
        return REGION_NAMES[name]
    for prefix,title in (("river_","牌河"),("meld_","副露"),("score_","分数")):
        if name.startswith(prefix):
            return {"self":"自己","right":"下家","top":"对家","left":"上家"}[name[len(prefix):]]+title
    return name


def meld_text(events):
    rows = []
    reverse = {v:k for k,v in KINDS.items()}
    for event in events:
        kind = event["type"]
        if kind == "ankan":
            rows.append("暗杠 " + " ".join(map(tile_name,event["consumed"])))
        else:
            source = " " + SEATS[event["target"]] if "target" in event else ""
            rows.append(reverse[kind]+source+" "+tile_name(event["pai"])+" | "+" ".join(map(tile_name,event["consumed"])))
    return "; ".join(rows)


def parse_melds(text,actor):
    result = []
    for row in text.replace("；",";").split(";"):
        if not row.strip():
            continue
        parts = row.strip().split()
        kind = KINDS.get(parts[0])
        if kind is None:
            raise ValueError("副露类型使用：吃、碰、明杠、暗杠、加杠。")
        if kind == "ankan":
            result.append({"type":kind,"actor":actor,"consumed":parse_tiles(" ".join(parts[1:]))})
        else:
            head,tail = row.split("|",1)
            words = head.split()
            event = {"type":kind,"actor":actor,"pai":parse_tiles(words[-1])[0],"consumed":parse_tiles(tail)}
            if kind != "kakan":
                event["target"] = SEATS.index(words[1])
            result.append(event)
    return result


class SnapshotReview(tk.Toplevel):
    def __init__(self,parent,response,callback):
        super().__init__(parent)
        self.response,self.callback = response,callback
        self.title("核对当前局面")
        self.geometry("850x790")
        self.transient(parent)
        self.grab_set()
        panel = ttk.Frame(self,padding=12); panel.pack(fill="both",expand=True)
        ttk.Label(panel,text="牌面用空格分隔，例如：1万 5筒赤 东 白。牌河立直牌后加 *。空区域留空。\n"
                  "副露例：碰 对家 2万 | 2万 2万；吃 上家 3索 | 4索 5索。获牌写在 | 前。",wraplength=815).pack(anchor="w")
        self.values = {}
        grid = ttk.Frame(panel); grid.pack(fill="both",expand=True,pady=8)
        grid.columnconfigure(1,weight=1)
        regions = response["observation"]["regions"]
        for row,(name,r) in enumerate((n,r) for n,r in regions.items() if r["kind"] not in {"buttons","choices"}):
            existing = response.get("corrections",{}).get(name)
            if existing is None:
                if r["kind"] == "text":
                    existing = r["text"]["value"]
                elif r["kind"] == "meld":
                    existing = []
                    for g in r.get("groups",[]):
                        called = g.get("called_positions",[])
                        if g.get("type") in {"chi","pon"} and len(called)==1 and g.get("target") is not None:
                            i=called[0]; existing.append({"type":g["type"],"actor":r["player"],"target":g["target"],
                                                         "pai":g["labels"][i],"consumed":[t for j,t in enumerate(g["labels"]) if j!=i]})
                else:
                    existing = [{"tile":i["label"],"riichi":i.get("horizontal") is True} for i in r["items"]] if r["kind"]=="river" else [i["label"] for i in r["items"]]
            if r["kind"] == "text":
                text = ("东" if existing["bakaze"]=="E" else "南")+str(existing["kyoku"]) if isinstance(existing,dict) else "" if existing is None else str(existing)
            elif r["kind"] == "meld":
                text = meld_text(existing)
            elif r["kind"] == "river":
                text = " ".join(tile_name(i if isinstance(i,str) else i["tile"])+("*" if isinstance(i,dict) and i.get("riichi") else "") for i in existing)
            else:
                text = " ".join(map(tile_name,existing))
            variable = tk.StringVar(value=text); self.values[name]=variable
            ttk.Label(grid,text=region_name(name)+( "（需核对）" if not r["complete"] else "")).grid(row=row,column=0,sticky="w",padx=5,pady=2)
            ttk.Entry(grid,textvariable=variable).grid(row=row,column=1,sticky="ew",padx=5,pady=2)
        controls = ttk.Frame(panel); controls.pack(fill="x")
        ownwind = response["observation"].get("self_wind",{}).get("value")
        dealer = (-"ESWN".index(ownwind))%4 if ownwind in {"E","S","W","N"} else 0
        self.dealer = tk.StringVar(value=SEATS[response.get("context",{}).get("oya",dealer)])
        ttk.Label(controls,text="当前庄家").pack(side="left")
        ttk.Combobox(controls,textvariable=self.dealer,values=SEATS,state="readonly",width=8).pack(side="left",padx=5)
        phase = response.get("snapshot",{}).get("phase") if response.get("snapshot") else None
        if phase is None:
            phase = "draw" if any(r["items"] for r in regions.values() if r["kind"]=="draw") else "reaction"
        self.phase = tk.StringVar(value={"draw":"摸牌后出牌","reaction":"响应他人弃牌","call_discard":"吃碰后出牌"}[phase])
        ttk.Combobox(controls,textvariable=self.phase,values=["摸牌后出牌","响应他人弃牌","吃碰后出牌"],state="readonly",width=15).pack(side="left",padx=5)
        target=response.get("context",{}).get("target")
        if target is None:
            target=response["observation"].get("reaction_discard",{}).get("target",3)
        self.target = tk.StringVar(value=SEATS[target])
        ttk.Label(controls,text="刚出牌者").pack(side="left")
        ttk.Combobox(controls,textvariable=self.target,values=SEATS[1:],state="readonly",width=7).pack(side="left",padx=5)
        self.timer = tk.BooleanVar(value=response["observation"].get("action_timer",{}).get("active",False))
        ttk.Checkbutton(panel,text="确认当前自己的行动计时器正在显示，画面稳定",variable=self.timer).pack(anchor="w",pady=6)
        self.declared = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel,text="自己已按立直，正在选择宣言弃牌",variable=self.declared).pack(anchor="w")
        detected={i["label"] for r in regions.values() if r["kind"]=="buttons"
                  for i in r["items"] if i.get("clickable") is True}
        self.actions = {label:tk.BooleanVar(value=label in detected) for label in ("chi","pon","kan","ron","tsumo","reach")}
        actions = ttk.Frame(panel); actions.pack(fill="x")
        for label,title in (("chi","吃"),("pon","碰"),("kan","杠"),("ron","荣和"),("tsumo","自摸"),("reach","立直")):
            ttk.Checkbutton(actions,text="可"+title,variable=self.actions[label]).pack(side="left")
        ttk.Button(panel,text="确认这些当前信息并获取建议",command=self.submit).pack(fill="x",pady=8)

    def submit(self):
        try:
            corrections = {}
            regions = self.response["observation"]["regions"]
            for name,var in self.values.items():
                text,r = var.get().strip(),regions[name]
                if r["kind"] == "text":
                    if r["field"]=="round":
                        if len(text)!=2 or text[0] not in "东南" or text[1] not in "1234":
                            raise ValueError("场次例如东1或南4。")
                        corrections[name]={"bakaze":"E" if text[0]=="东" else "S","kyoku":int(text[1])}
                    else:
                        corrections[name]=int(text)
                elif r["kind"] == "meld":
                    corrections[name]=parse_melds(text,r["player"])
                elif r["kind"] == "river":
                    corrections[name]=[{"tile":parse_tiles(t.rstrip("*"))[0],"riichi":t.endswith("*"),"tedashi":False} for t in text.split()]
                else:
                    corrections[name]=parse_tiles(text)
            phase={"摸牌后出牌":"draw","响应他人弃牌":"reaction","吃碰后出牌":"call_discard"}[self.phase.get()]
            context={"oya":SEATS.index(self.dealer.get()),"phase":phase,"timer_confirmed":self.timer.get()}
            if self.declared.get():
                own=[r["player"] for r in regions.values() if r["kind"]=="hand"][0]
                context["riichi_declared"]=[i==own for i in range(4)]
            if not self.timer.get():
                raise ValueError("需要确认当前自己的行动计时器正在显示。")
            if phase=="reaction":
                context.update(target=SEATS.index(self.target.get()),allowed_actions=["pass",*[k for k,v in self.actions.items() if v.get()]])
            if phase=="call_discard":
                own=[r["player"] for r in regions.values() if r["kind"]=="hand"][0]
                name=next(n for n,r in regions.items() if r["kind"]=="meld" and r["player"]==own)
                if corrections[name]:
                    context["last_call"]=corrections[name][-1]
            review={"image_sha256":self.response["image_sha256"],"corrections":corrections,"context":context}
            self.callback(review)
            self.destroy()
        except (ValueError,KeyError,IndexError) as exc:
            messagebox.showerror("核对信息不完整",str(exc),parent=self)


class AdvicePanel:
    def __init__(self,root,config_path,output,image_path=None):
        self.root,self.config_path,self.output = root,Path(config_path),Path(output)
        self.config=load_config(config_path)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="screenshot-advice")
        self.queue=queue.Queue(maxsize=2)
        self.stop_event=threading.Event()
        self.generation=0; self.closed=False; self.live=False; self.result=None; self.pixels=None
        self.service=None
        root.title("Majan · 雀魂助手"); root.geometry("560x640"); root.minsize(450,640)
        self.theme=IvoryTheme(root)
        header=tk.Frame(root,background=PAPER);header.pack(fill="x",padx=16,pady=(10,10))
        mark=tk.Canvas(header,width=29,height=35,background=PAPER,highlightthickness=0)
        mark.pack(side="left",padx=(0,10));tile(mark,1,0,23,29,'F')
        tk.Label(header,text="Majan 雀魂助手",background=PAPER,foreground=INK,
                 font=(self.theme.serif,-22,'bold')).pack(side="left")
        self.connection=tk.Label(header,text="● 未连接",background=PAPER,foreground=MUTED,font=(UI,-11))
        self.connection.pack(side="right")
        bar=tk.Frame(root,background=PAPER); bar.pack(fill="x",padx=16)
        settings=Card(bar,padding=9);settings.pack(fill="x")
        settings.body.columnconfigure(0,weight=1)
        tk.Label(settings.body,text="游戏窗口",background=CARD,foreground=MUTED,font=(UI,-11)).grid(row=0,column=0,sticky="w",pady=(0,2))
        tk.Label(settings.body,text="画面大小",background=CARD,foreground=MUTED,font=(UI,-11)).grid(row=0,column=1,sticky="w",padx=(12,0),pady=(0,2))
        self.window=tk.StringVar()
        self.selector=ttk.Combobox(settings.body,textvariable=self.window,state="readonly",width=20,
                                   style='Ivory.TCombobox',font=(UI,-13))
        self.selector.grid(row=1,column=0,sticky="ew")
        self.selector.bind('<<ComboboxSelected>>',lambda event:self.update_connection())
        self.client_size=tk.StringVar(value="1920×1080")
        self.size_selector=ttk.Combobox(settings.body,textvariable=self.client_size,values=list(CLIENT_PRESETS),
                                       state="readonly",width=12,style='Ivory.TCombobox',font=(UI,-13))
        self.size_selector.grid(row=1,column=1,sticky="ew",padx=(12,0))
        actions=tk.Frame(bar,background=PAPER); actions.pack(fill="x",pady=(8,0))
        for column in range(2):actions.columnconfigure(column,weight=1,uniform="action")
        self.buttons={}
        for index,(title,callback) in enumerate((("刷新雀魂窗口",self.refresh),("整理窗口",self.arrange),
                                                 ("开始自动建议",self.start),("刷新截图",self.capture_now))):
            button=ttk.Button(actions,text=title,command=callback,
                              style='Ivory.Primary.TButton' if title=='开始自动建议' else 'Ivory.TButton',
                              image=self.theme.icons[title],compound='left')
            button.grid(row=index//2,column=index%2,sticky="ew",padx=(4 if index%2 else 0,0 if index%2 else 4),
                        pady=(6 if index>=2 else 0,0))
            self.buttons[title]=button
        self.capture_button=self.buttons["刷新截图"]
        self.status=tk.StringVar(value="选择窗口与画面大小，开始自动建议。")
        self.status_label=tk.Label(root,textvariable=self.status,background=PAPER,foreground=MUTED,
                                   font=(UI,-11),wraplength=528,anchor='w',justify='left',height=1)
        self.status_label.pack(fill="x",padx=18,pady=(2,4))
        self.advice=tk.StringVar(value="尚无建议")
        self.hero=AdviceCard(root);self.hero.pack(fill="x",padx=16,pady=(0,10))
        self.advice.trace_add('write',lambda *_:self.hero.show(self.result,self.advice.get()))
        self.table=CandidateList(root);self.table.pack(fill="both",expand=True,padx=16,pady=(0,14))
        root.bind('<Configure>',self.resize_panel,add='+')
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.after(100,self.poll)
        self.refresh()
        if image_path:
            self.analyze_file(Path(image_path))

    def resize_panel(self,event):
        if event.widget is self.root:
            self.status_label.configure(wraplength=max(1,event.width-36))

    def update_connection(self):
        connected=0<=self.selector.current()<len(self.areas)
        self.connection.configure(text='● 已连接' if connected else '● 未连接',foreground=GREEN if connected else MUTED)

    def publish(self,generation,result,pixels):
        item=(generation,result,pixels)
        try:self.queue.put_nowait(item)
        except queue.Full:
            try:self.queue.get_nowait()
            except queue.Empty:pass
            self.queue.put_nowait(item)

    def get_service(self,stable_frames):
        if self.service is None:
            self.service=ScreenshotAdvice(self.config,stable_frames=stable_frames)
        else:
            self.service.stable_frames=stable_frames
        return self.service

    def refresh(self):
        try:
            self.areas=windows("雀魂")
            self.selector["values"]=[f"{a.title} · {a.width}×{a.height}" for a in self.areas]
            if self.areas:
                self.selector.current(0)
                size=f"{self.areas[0].width}×{self.areas[0].height}"
                if size in CLIENT_PRESETS:self.client_size.set(size)
            else:self.window.set('未找到雀魂窗口')
        except Exception as exc:self.status.set(str(exc)); self.areas=[]
        self.update_connection()

    def stop(self):
        self.stop_event.set(); self.generation+=1; self.live=False
        self.result=None; self.advice.set("已停止；尚无当前建议")
        self.table.clear()

    def start(self):
        index=self.selector.current()
        if index<0 or index>=len(self.areas):
            self.status.set("请先选择雀魂窗口。"); return
        self.stop()
        if self.arrange(stop=False) is None:
            return
        self.stop_event=threading.Event(); self.live=True
        generation,stop,hwnd=self.generation,self.stop_event,self.areas[index].hwnd
        self.status.set("正在准备识别与建议模型，完成后自动监测自己的行动…")
        self.advice.set("正在准备识别与建议模型…")
        self.executor.submit(self.live_worker,generation,stop,hwnd)

    def arrange(self,stop=True):
        index=self.selector.current()
        if index<0 or index>=len(self.areas):
            self.status.set("请先选择雀魂窗口。"); return
        if stop:self.stop()
        try:
            previous=self.areas[index]
            area,layout=arrange_client(previous.hwnd,CLIENT_PRESETS[self.client_size.get()],
                                       expected_pid=previous.pid)
            self.areas[index]=area
            self.selector["values"]=[f"{a.title} · {a.width}×{a.height}" for a in self.areas]
            self.selector.current(index)
            position_panel(self.root,layout)
            self.status.set(f"游戏画面已整理为 {area.width}×{area.height}，截图位置已自动对齐。")
            return area
        except Exception as exc:
            self.status.set(str(exc))
            return None

    def live_worker(self,generation,stop,hwnd,initial_edge=None,pending_start=False):
        capture=None
        try:
            service=self.get_service(1); service.clear()
            if hasattr(service,'prepare'):service.prepare()
            capture=WindowCapture(hwnd,self.config)
            capture.window_content_fallback=True
            directory=self.output/(datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-")+str(generation))
            edge=initial_edge or TimerEdge(); action_index=0; waiting_published=False; last_capture_issue=None
            while not stop.is_set():
                try:
                    active,seconds,marker=action_probe(capture,service,self.config)
                    if stop.is_set():break
                    event=edge.observe(active,seconds,marker)
                    if pending_start and active:event="start"
                    pending_start=False
                    if event == "start":
                        timing=AdviceTiming()
                        # The countdown's rising edge triggers one saved board
                        # analysis; the window stream itself is not recorded.
                        service.clear(); waiting_published=False; action_index+=1
                        self.publish(generation,{"status":"confirming_frame","reason":"正在获取当前行动的截图…",
                                                 "recommendation":None,"observation":None},None)
                        # Let the draw/call animation settle before the one snapshot.
                        # Event.wait keeps Stop/Close responsive during the delay.
                        if stop.wait(CAPTURE_DELAY_SECONDS) or stop.is_set():break
                        (active,seconds,marker),event=checked_action_probe(capture,service,self.config,edge,stop)
                        if stop.is_set():break
                        if not active:
                            edge.cancel()
                            result={"status":"waiting_turn","reason":"等待截图期间行动已结束，已取消本次截图。","recommendation":None,"observation":None}
                            write_json(self.output/"latest.json",result)
                            self.publish(generation,result,None)
                            waiting_published=True
                            stop.wait(.15)
                            continue
                        frame=timing.grab(capture)
                        frame_marker=service.action_marker(frame.pixels) if hasattr(service,'action_marker') else marker
                        edge.bind(frame_marker,seconds)
                        ticket=edge.serial
                        result=timing.process(service,frame)
                        if stop.is_set():break
                        def same_action():
                            nonlocal pending_start
                            (active,_,_),event=checked_action_probe(capture,service,self.config,edge,stop)
                            if event=="start":pending_start=True
                            if not active:edge.cancel()
                            return active and edge.serial==ticket
                        # Incomplete board while the turn is still ours: animations
                        # and passing covers clear within a fraction of a second.
                        # Show the first reading at once; recapturing takes up
                        # to a few seconds and the player may act meanwhile.
                        current=same_action()
                        if stop.is_set():break
                        if current and result.get("status")=="needs_confirmation" and recapture_gaps(result):
                            # Not a request to review yet: arrows, highlights and
                            # call animations usually clear on the next frame.
                            self.publish(generation,{**result,"status":"confirming_frame",
                                                     "reason":"画面还在变化（"+"、".join(recapture_reasons(result))+"），正在补截确认……"},frame.pixels)
                        if current and result.get("status")=="needs_confirmation" and recapture_gaps(result):
                            frame,result,_=recapture_until_complete(
                                frame,result,grab=lambda:timing.grab(capture),
                                process=lambda shot:timing.process(service,shot),
                                timer_active=same_action,wait=stop.wait,
                                deadline=timing.started+ADVICE_RESPONSE_SECONDS-.25)
                            current=same_action()
                        if stop.is_set():break
                        if not current:
                            service.clear()
                            result={"status":"confirming_frame" if pending_start else "waiting_turn",
                                    "reason":"行动已变化，正在获取新局面…" if pending_start else "本次行动已结束，已清除建议。",
                                    "recommendation":None,"observation":None}
                            waiting_published=not pending_start
                        if stop.is_set():break
                        frame.metadata.update(capture_trigger="automatic_action",action_serial=ticket)
                        timing.stamp(result)
                        # Showing a checked current result must not wait for PNG
                        # encoding, overlays and evidence files to reach disk.
                        self.publish(generation,result,frame.pixels)
                        image_path=frame.save(directory/f'action-{action_index:04d}')
                        if result.get('observation') is not None:
                            result["observation"]["frame"]=deepcopy(frame.metadata)
                            save_advice(image_path.parent/"advice.json",result,frame.pixels)
                        else:
                            write_json(image_path.parent/"advice.json",result)
                        write_json(self.output/"latest.json",result)
                    elif event == "end" or (not edge.visible and not waiting_published):
                        service.clear()
                        if not waiting_published:
                            result={"status":"waiting_turn","reason":"等待自己的行动计时器。","recommendation":None,"observation":None}
                            write_json(self.output/"latest.json",result)
                            self.publish(generation,result,None)
                            waiting_published=True
                    stop.wait(.15)
                except CaptureError as exc:
                    # A moved, covered or briefly unavailable window must not end
                    # the session: report it once and keep trying.
                    if stop.is_set():break
                    service.clear(); edge=TimerEdge()
                    pending_start=False
                    reason=f"{exc} 正在重试，恢复后自动继续。"
                    if reason!=last_capture_issue:
                        last_capture_issue=reason
                        result={"status":"waiting_turn","reason":reason,"recommendation":None,"observation":None}
                        write_json(self.output/"latest.json",result)
                        self.publish(generation,result,None)
                    waiting_published=False
                    stop.wait(CAPTURE_RETRY_SECONDS)
                    continue
                last_capture_issue=None
        except Exception as exc:
            if not stop.is_set():
                result={"status":"error","reason":str(exc),"recommendation":None}
                write_json(self.output/"latest.json",result)
                self.publish(generation,result,None)
        finally:
            if capture:capture.close()

    def open_image(self):
        path=filedialog.askopenfilename(filetypes=[("截图","*.png *.jpg *.jpeg")])
        if path:self.analyze_file(Path(path))

    def capture_now(self):
        """Take a fresh selected-window frame; continue live monitoring afterward."""
        index=self.selector.current()
        if index<0 or index>=len(self.areas):
            self.status.set("请先选择雀魂窗口。");return
        resume=self.live
        hwnd=self.areas[index].hwnd
        self.stop()
        self.stop_event=threading.Event()
        self.live=resume
        generation,stop=self.generation,self.stop_event
        self.status.set("正在重新截取雀魂窗口并分析…")
        self.advice.set("正在获取当前截图并分析…")
        self.executor.submit(self.capture_worker,generation,stop,hwnd,resume)

    def capture_worker(self,generation,stop,hwnd,resume=False):
        capture=None;edge=TimerEdge();pending_start=False
        timing=AdviceTiming()
        try:
            if stop.is_set() or generation!=self.generation:return
            service=self.get_service(1);service.clear()
            capture=WindowCapture(hwnd,self.config)
            capture.window_content_fallback=True
            frame=timing.grab(capture)
            if stop.is_set() or generation!=self.generation:return
            active=service.timer_active(frame.pixels,cropped=False)
            seconds=getattr(service,'timer_seconds',None)
            marker=service.action_marker(frame.pixels) if hasattr(service,'action_marker') else None
            edge.observe(active,seconds,marker)
            ticket=edge.serial
            result=timing.process(service,frame)
            if stop.is_set() or generation!=self.generation:return
            (current,_,_),event=checked_action_probe(capture,service,self.config,edge,stop)
            if stop.is_set() or generation!=self.generation:return
            pending_start=event=='start'
            if not current:edge.cancel()
            if (active and not current) or edge.serial!=ticket:
                service.clear()
                result={"status":"waiting_turn","reason":"截图后行动已变化，已清除旧建议。"+
                        ("自动建议将继续获取新局面。" if resume else "请点击“刷新截图”重新取图。"),
                        "recommendation":None,"observation":None}
            frame.metadata.update(capture_trigger="manual_screenshot_button",action_serial=ticket)
            timing.stamp(result)
            self.publish(generation,result,frame.pixels)
            directory=self.output/(datetime.now(timezone.utc).strftime("manual-%Y%m%dT%H%M%S-")+str(generation))
            frame.save(directory)
            if result.get('observation') is not None:
                result['observation']['frame']=deepcopy(frame.metadata)
                save_advice(directory/'advice.json',result,frame.pixels)
            else:write_json(directory/'advice.json',result)
            write_json(self.output/'latest.json',result)
        except Exception as exc:
            if not stop.is_set() and generation==self.generation:
                result={"status":"error","reason":str(exc),"recommendation":None,"observation":None}
                write_json(self.output/'latest.json',result)
                self.publish(generation,result,None)
        finally:
            if capture:capture.close()
        if resume and not stop.is_set() and generation==self.generation:
            self.live_worker(generation,stop,hwnd,initial_edge=edge,pending_start=pending_start)

    def analyze_file(self,path,review=None):
        self.stop(); generation=self.generation
        self.status.set("正在读取当前截图并获取建议…")
        self.advice.set("正在读取当前截图并获取建议…")
        def worker():
            try:
                pixels,metadata=load_screenshot(path,self.config)
                service=self.get_service(1); service.clear()
                result=service.process(pixels,metadata,review)
                if generation != self.generation:return
                save_advice(self.output/"latest.json",result,pixels)
                self.publish(generation,result,pixels)
            except Exception as exc:self.publish(generation,{"status":"error","reason":str(exc),"recommendation":None},None)
        self.executor.submit(worker)

    def poll(self):
        if self.closed:return
        try:
            while True:
                generation,result,pixels=self.queue.get_nowait()
                if generation != self.generation:continue
                self.result=result
                if pixels is not None:
                    self.pixels=pixels
                self.advice.set(panel_message(format_advice(result)))
                timer=(result.get("observation") or {}).get("action_timer",{}).get("value")
                timer_text = (str(timer['fixed_seconds']) +
                              (f" +{timer['reserve_seconds']}" if timer['reserve_seconds'] is not None else "")) if timer else ""
                latency=result.get('latency')
                elapsed=(latency['trigger_to_result_ms']+
                         max(0,time.perf_counter_ns()-latency['published_monotonic_ns'])/1e6
                         if latency else result.get('total_ms',0))
                self.status.set((f"自己的计时器：{timer_text}；" if timer else "")+
                                (f"当前建议已更新 · 用时 {elapsed/1000:.1f} 秒" if result["status"]=="ready" else panel_message(result.get("reason") or result["status"])))
                rows=[]
                if result["status"]=="ready":
                    for candidate,percentage in candidate_percentages(result["decision"]["legal_candidates"]):
                        rec={**candidate,"event":candidate["variants"][0]}
                        rows.append((format_advice({"status":"ready","recommendation":rec}),percentage))
                self.table.set_rows(rows)
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def review(self):
        if not self.result or not self.result.get("observation"):
            self.status.set("请先打开一张截图或等待一次行动截图。"); return
        response=deepcopy(self.result)
        # Pause live processing while the human checks this exact frame.
        pixels=self.pixels.copy(); self.stop()
        path=self.output/("review-"+response["request_id"]+".png")
        from app.capture.frames import save_image
        save_image(path,pixels)
        def apply(review):
            write_json(path.with_suffix(".review.json"),review)
            self.analyze_file(path,review)
        SnapshotReview(self.root,response,apply)

    def close(self):
        self.closed=True; self.stop_event.set(); self.generation+=1
        self.executor.shutdown(wait=False,cancel_futures=True); self.root.destroy()


def launch(config_path,output,image_path=None):
    root=tk.Tk(); AdvicePanel(root,config_path,output,image_path); root.mainloop()
