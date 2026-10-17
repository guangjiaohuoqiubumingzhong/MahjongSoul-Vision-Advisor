from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import uuid

from PIL import Image, ImageTk

from app.capture.frames import config_digest, load_image, save_image
from app.state.events import TILES
from app.vision.config import load_config, pixel_rect, validate_config, write_json
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


class Workbench:
    def __init__(self, root: tk.Tk, image_path: Path, config_path: Path):
        self.root, self.image_path, self.config_path = root, image_path.resolve(), config_path.resolve()
        self.config = load_config(config_path)
        self.pixels = load_image(image_path)
        frame_path = self.image_path.with_suffix(".frame.json")
        self.frame_metadata = json.loads(frame_path.read_text(encoding="utf-8-sig")) if frame_path.exists() else {}
        self.frame_metadata["image_path"] = str(self.image_path)
        if [self.pixels.shape[1], self.pixels.shape[0]] != self.config["client_size"]:
            raise ValueError("图像尺寸与窗口配置不一致。")
        self.annotation_path = self.image_path.with_suffix(".annotation.json")
        self.annotation = {"schema_version": 1, "image": str(self.image_path), "game_id": "", "split": "train",
                           "reviewed": False, "frame_quality": "stable", "complete_regions": [], "targets": [],
                           "information_scope": "self_hand_and_public_information_only"}
        if self.annotation_path.exists():
            self.annotation = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.observation, self.selected, self.drag_start = None, None, None
        self.pending_corrections = []
        root.title("Majan · 截图校准与人工核对")
        root.geometry("1320x880")
        tools = ttk.Frame(root, padding=8)
        tools.pack(fill="x")
        for text, callback in (("打开截图", self.open_image), ("识别与带框预览", self.recognize),
                               ("获取当前出牌建议", self.open_advice),
                               ("导出结果", self.export_result), ("保存标注", self.save_annotation),
                               ("保存区域配置", self.save_config)):
            ttk.Button(tools, text=text, command=callback).pack(side="left", padx=3)
        self.message = tk.StringVar(value="拖框建立牌面标注；点击现有框可修正。识别分数不是校准后的概率。")
        ttk.Label(root, textvariable=self.message, padding=8).pack(fill="x")
        content = ttk.Frame(root)
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, background="#1d2431", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        panel_holder = ttk.Frame(content)
        panel_holder.pack(side="right", fill="y")
        panel_canvas = tk.Canvas(panel_holder, width=255, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel_holder, orient="vertical", command=panel_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        panel_canvas.pack(side="left", fill="y")
        panel_canvas.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(panel_canvas, padding=10)
        panel_item = panel_canvas.create_window(0, 0, window=panel, anchor="nw", width=255)
        panel.bind("<Configure>", lambda _: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
        panel_canvas.bind("<Configure>", lambda event: panel_canvas.itemconfigure(panel_item, width=event.width))
        self.mode = tk.StringVar(value="标注目标")
        self.region = tk.StringVar(value="hand")
        self.label = tk.StringVar(value="?")
        self.kind = tk.StringVar(value="tile")
        self.rotation = tk.IntVar(value=0)
        self.game = tk.StringVar(value=self.annotation["game_id"])
        self.split = tk.StringVar(value=self.annotation["split"])
        self.quality = tk.StringVar(value=self.annotation.get("frame_quality", "stable"))
        self.reviewed = tk.BooleanVar(value=self.annotation["reviewed"])
        self.group = tk.StringVar(value="")
        self.meld_type = tk.StringVar(value="")
        self.slot = tk.StringVar(value="")
        self.source = tk.StringVar(value="")
        self.button_state = tk.StringVar(value="unknown")
        controls = [("工作模式", self.mode, ("标注目标", "调整区域")), ("区域", self.region, tuple(self.config["regions"])),
                    ("目标类型", self.kind, ("tile", "back", "text", "button")),
                    ("牌面 / 原文 / 按钮名称", self.label, tuple(["?", *TILES, "back", "chi", "pon", "kan", "reach", "ron", "tsumo", "pass"])),
                    ("逆时针转正角度", self.rotation, (0, 90, 180, 270)), ("对局编号", self.game, None),
                    ("数据划分（按对局）", self.split, ("train", "debug", "test")),
                    ("画面质量", self.quality, ("stable", "animation", "occluded", "low_resolution", "unknown")),
                    ("副露组编号", self.group, None), ("副露类型", self.meld_type, ("", "chi", "pon", "daiminkan", "ankan", "kakan")), ("牌河槽 行,列", self.slot, None),
                    ("副露来源玩家 0–3", self.source, None), ("按钮状态", self.button_state, ("enabled", "disabled", "clicked", "closed", "unknown"))]
        for text, variable, values in controls:
            ttk.Label(panel, text=text).pack(anchor="w", pady=(5, 1))
            widget = ttk.Combobox(panel, textvariable=variable, values=values) if values else ttk.Entry(panel, textvariable=variable)
            widget.pack(fill="x")
        ttk.Checkbutton(panel, text="本图标注已人工核对", variable=self.reviewed).pack(anchor="w", pady=8)
        ttk.Button(panel, text="应用修正到选中框", command=self.apply_correction).pack(fill="x", pady=3)
        ttk.Button(panel, text="删除选中标注", command=self.delete_target).pack(fill="x", pady=3)
        ttk.Button(panel, text="核对区域完整性…", command=self.mark_regions).pack(fill="x", pady=3)
        ttk.Button(panel, text="核对暂停证据区域…", command=self.mark_pause_regions).pack(fill="x", pady=3)
        ttk.Button(panel, text="标记所有区域已校准", command=self.mark_calibrated).pack(fill="x", pady=3)
        self.details = tk.Text(panel, height=9, width=27, wrap="word")
        self.details.pack(fill="both", expand=True, pady=6)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<B1-Motion>", self.pointer_move)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.render()

    def render(self):
        width, height = self.pixels.shape[1], self.pixels.shape[0]
        self.scale = min(max(1, self.canvas.winfo_width())/width, max(1, self.canvas.winfo_height())/height)
        size = (max(1, round(width*self.scale)), max(1, round(height*self.scale)))
        pixels = self.pixels
        self.photo = ImageTk.PhotoImage(Image.fromarray(pixels[:, :, ::-1]).resize(size, Image.Resampling.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        for name, region in self.config["regions"].items():
            self.draw_box(pixel_rect(region["rect"], width, height), "#eab54b", name, 1)
        if self.observation:
            for region in self.observation["regions"].values():
                for item in region["items"]:
                    self.draw_box(item["bbox"], "#29e094" if item["label"] != "?" else "#ff7566", item["label"], 2)
                for item in region.get("boundary_fragments", []):
                    self.draw_box(item["bbox"], "#ff7566", "clipped ?", 2)
                for item in region.get("unresolved_components", []):
                    self.draw_box(item["bbox"], "#ff7566", "foreground ?", 2)
        for index, target in enumerate(self.annotation["targets"]):
            self.draw_box(target["bbox"], "#ffffff" if index == self.selected else "#29e094", target["label"], 2)

    def draw_box(self, box, color, label, thickness):
        x, y, w, h = [v*self.scale for v in box]
        self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=thickness)
        self.canvas.create_text(x+2, y+2, anchor="nw", text=label, fill=color, font=("Segoe UI", 9))

    def point(self, event):
        return [min(self.pixels.shape[1]-1, max(0, round(event.x/self.scale))),
                min(self.pixels.shape[0]-1, max(0, round(event.y/self.scale)))]

    def pointer_down(self, event):
        self.drag_start = self.point(event)

    def pointer_move(self, event):
        if self.drag_start:
            self.render()
            x, y = self.drag_start
            right, bottom = self.point(event)
            self.draw_box([min(x,right), min(y,bottom), abs(right-x), abs(bottom-y)], "#ffffff", "", 2)

    def pointer_up(self, event):
        if self.drag_start is None:
            return
        start, end = self.drag_start, self.point(event)
        self.drag_start = None
        x, y = min(start[0], end[0]), min(start[1], end[1])
        w, h = abs(end[0]-start[0]), abs(end[1]-start[1])
        if w < 4 or h < 4:
            matches = [index for index, item in enumerate(self.annotation["targets"]) if item["bbox"][0] <= x <= item["bbox"][0]+item["bbox"][2] and item["bbox"][1] <= y <= item["bbox"][1]+item["bbox"][3]]
            if not matches and self.observation:
                proposed = []
                for name, region in self.observation["regions"].items():
                    candidates = [*region["items"],*region.get("boundary_fragments",[]),*region.get("unresolved_components",[])]
                    if region["kind"] == "text":
                        candidates = [{**region["text"], "bbox":region["bbox"], "label":region["text"]["raw_text"]}]
                    for item in candidates:
                        bx,by,bw,bh = item["bbox"]
                        if bx <= x <= bx+bw and by <= y <= by+bh:
                            proposed.append((bw*bh,name,region,item))
                if proposed:
                    _,name,region,item = min(proposed, key=lambda p:p[0])
                    target = {"id":uuid.uuid4().hex[:12],"region":name,"label":item["label"],"bbox":item["bbox"],
                              "kind":"text" if region["kind"] == "text" else "button" if region["kind"] == "buttons" else "tile",
                              "player":region["player"],"rotation":(region["rotation"]+item.get("face_rotation",0))%360,"source":"recognition_needs_manual_review",
                              "original_recognition":deepcopy(item)}
                    if "state" in item:
                        target["state"] = item["state"]
                    if "slot" in item:
                        target["slot"] = item["slot"]
                    if "slot_index" in item:
                        target["slot_index"] = item["slot_index"]
                    self.annotation["targets"].append(target)
                    self.reviewed.set(False)
                    matches = [len(self.annotation["targets"])-1]
            self.selected = matches[-1] if matches else None
            if self.selected is not None:
                target = self.annotation["targets"][self.selected]
                self.region.set(target["region"])
                self.label.set(target["label"])
                self.kind.set(target.get("kind", "tile"))
                self.rotation.set(target.get("rotation", 0))
                self.group.set(target.get("group_id", ""))
                self.meld_type.set(target.get("meld_type") or "")
                self.slot.set(",".join(map(str, target["slot"])) if "slot" in target else "")
                self.source.set("" if target.get("target") is None else str(target["target"]))
                self.button_state.set(target.get("state", "unknown"))
                self.details.delete("1.0", "end")
                self.details.insert("end", json.dumps(target, ensure_ascii=False, indent=2))
        elif self.mode.get() == "调整区域":
            height, width = self.pixels.shape[:2]
            self.config["regions"][self.region.get()]["rect"] = [x/width, y/height, w/width, h/height]
            self.config["regions"][self.region.get()]["rotation"] = self.rotation.get()
            self.config["regions"][self.region.get()].pop("slots", None)
            self.config["regions"][self.region.get()].pop("grid", None)
            for name,region in self.config["regions"].items():
                if name == self.region.get() or region.get("hand_layout",{}).get("draw_region") == self.region.get():
                    region.pop("hand_layout",None)
            self.config["calibrated"] = False
        else:
            self.annotation["targets"].append(self.target_values([x,y,w,h]))
            self.selected = len(self.annotation["targets"])-1
            self.reviewed.set(False)
        self.render()

    def target_values(self, box):
        region = self.config["regions"][self.region.get()]
        target = {"id": uuid.uuid4().hex[:12], "bbox": box, "label": self.label.get(), "kind": self.kind.get(),
                  "region": self.region.get(), "player": region.get("player"), "rotation": self.rotation.get(),
                  "roles": [self.region.get()]}
        if self.kind.get() == "text":
            from app.vision.text import parse_text
            target["value"] = parse_text(self.label.get(), region.get("field", ""))
        if self.group.get():
            target["group_id"] = self.group.get()
        if self.meld_type.get():
            target["meld_type"] = self.meld_type.get()
        if self.source.get():
            target["target"] = int(self.source.get())
            if target["target"] not in range(4):
                raise ValueError("来源玩家编号无效。")
        if self.slot.get():
            target["slot"] = [int(part) for part in self.slot.get().split(",")]
            if len(target["slot"]) != 2:
                raise ValueError("槽位置格式为 行,列。")
        if self.kind.get() == "button":
            target["state"] = self.button_state.get()
        return target

    def apply_correction(self):
        if self.selected is None:
            return
        try:
            before = deepcopy(self.annotation["targets"][self.selected])
            after = self.target_values(before["bbox"])
            after["id"] = before["id"]
            self.annotation["targets"][self.selected] = after
            self.pending_corrections.append({"at": datetime.now(timezone.utc).isoformat(), "image": str(self.image_path), "before": before, "after": after})
            self.reviewed.set(False)
            self.render()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("修正失败", str(exc))

    def delete_target(self):
        if self.selected is not None:
            before = self.annotation["targets"].pop(self.selected)
            self.pending_corrections.append({"at": datetime.now(timezone.utc).isoformat(), "before": before, "after": None})
            self.selected = None
            self.reviewed.set(False)
            self.render()

    def mark_regions(self):
        self._mark_region_evidence("complete_regions","已完整核对区域",
                                   "输入已完整标注所有目标（含空区域）的区域名，以逗号分隔：")

    def mark_pause_regions(self):
        self._mark_region_evidence("expected_pause_regions","不确定画面的暂停证据",
                                   "输入动画或遮挡使信息无法确认的区域名，以逗号分隔；这些区域必须报告识别失败：")

    def _mark_region_evidence(self, field, title, prompt):
        value = simpledialog.askstring(title,prompt,initialvalue=",".join(self.annotation.get(field,[])))
        if value is not None:
            names = list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
            if not set(names) <= set(self.config["regions"]):
                messagebox.showerror("区域名无效", "存在配置以外的区域。")
                return
            before = deepcopy(self.annotation.get(field,[]))
            if before != names:
                self.annotation[field] = names
                self.pending_corrections.append({"at":datetime.now(timezone.utc).isoformat(),"image":str(self.image_path),
                                                 "field":field,"before":before,"after":names})
                self.reviewed.set(False)
            self.message.set("区域证据已更新；核对后保存标注。")

    def mark_calibrated(self):
        self.config["calibrated"] = True
        self.message.set("区域标记为已校准；仍需独立对局数据通过阶段二验收。")

    def save_annotation(self):
        if not self.game.get().strip():
            messagebox.showerror("缺少对局编号", "同一对局的所有帧必须用同一个编号，避免验收数据泄漏。")
            return
        self.annotation.update({"game_id": self.game.get().strip(), "split": self.split.get(), "reviewed": self.reviewed.get(),
                                "frame_quality": self.quality.get(), "config_version": self.config["version"]})
        write_json(self.annotation_path, self.annotation)
        with self.annotation_path.with_suffix(".corrections.jsonl").open("a", encoding="utf-8") as stream:
            for correction in self.pending_corrections:
                stream.write(json.dumps(correction, ensure_ascii=False) + "\n")
        self.pending_corrections.clear()
        self.message.set(f"已保存：{self.annotation_path.name}")

    def save_config(self):
        validate_config(self.config)
        self.config["version"] = "steam-calibration-" + uuid.uuid4().hex[:8]
        write_json(self.config_path, self.config)
        self.message.set(f"已保存配置版本：{self.config['version']}")

    def recognize(self):
        try:
            key = config_digest(self.config)
            if getattr(self, "_recognizer_key", None) != key:
                self._recognizer = StaticRecognizer(self.config, TemplateBank(self.config["templates"],self.config.get("classifier_manifest")), TextReader())
                self._recognizer_key = key
            self.observation = self._recognizer.recognize(self.pixels, deepcopy(self.frame_metadata))
            self.render()
            self.details.delete("1.0", "end")
            self.details.insert("end", json.dumps(self.observation["issues"], ensure_ascii=False, indent=2))
            status = "识别待确认" if self.observation["status"] == "needs_confirmation" else "单帧已识别，等待跨帧确认"
            classifier = self.observation.get("tile_classifier", {}).get("name", "模板")
            self.message.set(f"状态：{status}；牌面识别：{classifier}；耗时 {self.observation['recognition_ms']:.1f} ms；点击错误框可修正。")
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))

    def export_result(self):
        if self.observation is None:
            self.recognize()
        if self.observation:
            path = self.image_path.with_suffix(".recognition.json")
            write_json(path, self.observation)
            save_image(path.with_suffix(".overlay.png"), overlay(self.pixels, self.observation))
            self.message.set(f"结果和带框图片已导出：{path.name}")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("截图", "*.png *.jpg *.jpeg")])
        if path:
            window = tk.Toplevel(self.root)
            Workbench(window, Path(path), self.config_path)

    def open_advice(self):
        from app.engine.runtime import ROOT
        from app.ui.advice import AdvicePanel
        output = ROOT / "reports/screenshot-advice"
        path = output / ("workbench-config-" + config_digest(self.config) + ".json")
        write_json(path,self.config)
        AdvicePanel(tk.Toplevel(self.root),path,output,self.image_path)


def launch(image_path: str | Path, config_path: str | Path):
    root = tk.Tk()
    Workbench(root, Path(image_path), Path(config_path))
    root.mainloop()
