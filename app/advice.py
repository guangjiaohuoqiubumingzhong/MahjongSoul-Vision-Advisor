"""Screenshot -> visible snapshot -> fresh Mortal decision, no history files."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import uuid

from app.capture.frames import load_image, save_image
from app.state.snapshot import build_snapshot, digest, SnapshotError
from app.vision.config import write_json
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader
from app.vision.turn import detect_timer, detect_timer_crop


def load_screenshot(path, config):
    from app.vision.config import scaled_client
    pixels = load_image(path)
    metadata_path = Path(path).with_suffix(".frame.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    height, width = pixels.shape[:2]
    if (scaled_client(width,height,config,allow_smaller=True) is None
            and scaled_client(width-2,height-32,config,allow_smaller=True) is not None):
        # Supplied Windows screenshot has a 31px title bar and 1px border.
        pixels = pixels[31:height-1,1:width-1].copy()
        metadata["window_crop"] = [1,31,width-2,height-32]
    metadata["image_path"] = str(Path(path).resolve())
    return pixels,metadata


class ScreenshotAdvice:
    def __init__(self, config, *, recognizer=None, backend=None, stable_frames=1):
        self.config = deepcopy(config)
        self.recognizer = recognizer or StaticRecognizer(config,TemplateBank(
            config["templates"],config.get("classifier_manifest")),TextReader())
        self.backend = backend
        self.stable_frames = stable_frames
        if type(stable_frames) is not int or stable_frames < 1:
            raise ValueError("稳定帧数应为正整数。")
        self.pending_key,self.pending_count,self.cached = None,0,None
        self.lock = threading.RLock()

    def prepare(self):
        """Load and warm models before the live loop waits for a player action."""
        with self.lock:
            if self.backend is None:
                from app.engine.runtime import OnnxRuntime
                from app.engine.snapshot import SnapshotBackend
                self.backend = SnapshotBackend(OnnxRuntime())
            model=getattr(getattr(self.recognizer,'bank',None),'model',None)
            if model is not None and hasattr(model,'warmup'):model.warmup()
            reader=getattr(self.recognizer,'text_reader',None)
            if reader is not None and hasattr(reader,'warmup'):reader.warmup()

    def process(self,pixels,metadata=None,review=None):
        with self.lock:
            started = time.perf_counter()
            observation = self.recognizer.recognize(pixels,metadata)
            response = {"schema_version":1,"request_id":uuid.uuid4().hex,
                        "image_sha256":observation["image_sha256"],
                        "status":"needs_confirmation","recommendation":None,
                        "observation":observation,"snapshot":None,"decision":None,
                        "corrections":{},"context":{},"reason":None}
            try:
                if review is not None:
                    if review.get("image_sha256") != observation["image_sha256"]:
                        raise SnapshotError("核对记录与当前截图不匹配，不能使用旧修正。")
                    response["corrections"] = deepcopy(review.get("corrections",{}))
                    response["context"] = deepcopy(review.get("context",{}))
                timer = observation.get("action_timer",{})
                if not timer.get("active") and response["context"].get("timer_confirmed") is not True:
                    self.clear()
                    response.update(status="waiting_turn",reason="等待自己的行动计时器。")
                    return response
                # After a call with several melds, the rules engine (and the
                # greyed 食替 tiles) settle which meld was just completed.
                snapshot,allowed = build_snapshot(observation,self.config,response["corrections"],response["context"],
                                                  legal_ids=self.legal_ids)
                response["snapshot"] = snapshot
                key = digest({"snapshot":snapshot,"allowed":allowed})
                if key != self.pending_key:
                    self.pending_key,self.pending_count,self.cached = key,1,None
                else:
                    self.pending_count += 1
                if self.pending_count < self.stable_frames:
                    response.update(status="confirming_frame",reason="正在确认当前局面稳定。")
                    return response
                if self.cached is None:
                    self.prepare()
                    self.cached = self.backend.decide(snapshot,allowed)
                check_reaction_banners(observation,snapshot,self.cached)
                response.update(status="ready",decision=deepcopy(self.cached),
                                recommendation=deepcopy(self.cached["recommendation"]))
                return response
            except (SnapshotError,ValueError,KeyError,TypeError) as exc:
                self.clear()
                response["reason"] = str(exc)
                return response
            finally:
                response["total_ms"] = (time.perf_counter()-started)*1000

    def clear(self):
        self.pending_key,self.pending_count,self.cached = None,0,None

    def legal_ids(self,snapshot):
        """Rules-engine probe for build_snapshot; loads the backend on first use."""
        self.prepare()
        probe = getattr(self.backend,"legal_ids",None)
        return probe(snapshot) if probe else None

    # Total seconds of the last countdown read (fixed + reserve), or None.
    timer_seconds = None

    def timer_active(self,pixels,*,cropped=False):
        detect = detect_timer_crop if cropped else detect_timer
        reading = detect(pixels,self.config.get("action_timer"),self.recognizer.text_reader)
        value = reading.get("value") if reading.get("active") else None
        self.timer_seconds = (value["fixed_seconds"]+(value.get("reserve_seconds") or 0)) if value else None
        return reading.get("active",False)

    def action_marker(self,pixels,*,cropped=False,river_pixels=None):
        """Read hand/draw bodies and own-river count without running the network."""
        from app.vision.trigger import hand_action_marker,river_action_count
        from app.vision.config import to_reference,pixel_rect
        region = next((r for r in self.config.get('regions',{}).values()
                       if r['kind']=='hand' and 'hand_layout' in r
                       and r.get('player') == self.config['seat_mapping']['self']),None)
        if region is None:
            return None
        river=next((r for r in self.config.get('regions',{}).values()
                    if r['kind']=='river' and 'grid' in r
                    and r.get('player') == self.config['seat_mapping']['self']),None)
        if not cropped:
            reference,_,_=to_reference(pixels,self.config)
            x,y,w,h=pixel_rect(region['rect'],*self.config['client_size'])
            pixels=reference[y:y+h,x:x+w]
            if river:
                x,y,w,h=pixel_rect(river['rect'],*self.config['client_size'])
                river_pixels=reference[y:y+h,x:x+w]
        hand=hand_action_marker(pixels,region)
        if hand is None:return None
        count=river_action_count(river_pixels,river) if river and river_pixels is not None else None
        return (*hand,count)


# Mortal action ids of reaction calls (rule_mask indices).
REACTION_ACTION_IDS = {"chi": (38, 39, 40), "pon": (41,), "kan": (42,), "ron": (43,)}
BANNER_NAMES = {"chi": "吃", "pon": "碰", "kan": "杠", "ron": "和", "call": "未知鸣牌"}


def check_reaction_banners(observation,snapshot,decision):
    """Reaction banners read by colour must match the calls the rules allow.

    Each legal call kind shows exactly one banner (one 吃 for any number of
    chi patterns). A difference, such as a ron the engine allows but the game
    does not offer (furiten it cannot see), stops the advice instead of
    recommending an action that is not on screen.
    """
    if snapshot.get("phase") != "reaction":
        return
    banners = [item["label"] for region in observation["regions"].values() if region["kind"] == "buttons"
               for item in region["items"] if item.get("evidence") == "reaction_banner_colour"]
    if not banners:
        return
    mask = decision.get("rule_mask") or []
    legal = sorted(kind for kind,ids in REACTION_ACTION_IDS.items()
                   if any(i < len(mask) and mask[i] for i in ids))
    shown = sorted("kan" if label == "call" else label for label in banners)
    if shown != legal:
        names = lambda kinds: "、".join(BANNER_NAMES[k] for k in kinds) or "无"
        raise SnapshotError(f"画面上的操作按钮（{names(sorted(banners))}）与规则推算的可选操作"
                            f"（{names(legal)}）不一致，请核对当前可用操作。")


# Board regions whose incompleteness a fresh frame can fix (animations, a
# window passing over the game). Buttons/text are not table evidence.
TABLE_KINDS = {"hand", "draw", "river", "meld", "dora"}
RECAPTURE_DELAY_SECONDS = .4
MAX_RECAPTURES = 3


def incomplete_table_regions(result):
    observation = result.get("observation") or {}
    return [name for name, region in (observation.get("regions") or {}).items()
            if region.get("kind") in TABLE_KINDS and region.get("complete") is False]


def reaction_source_missing(result):
    """Call buttons are on screen but the glowing last discard was not found.

    The outline pulses; at its dim phase it is invisible, so a fresh frame a
    moment later usually shows it.
    """
    observation = result.get("observation") or {}
    calls = any(item.get("clickable") is True and item.get("label") in {"chi", "pon", "kan", "ron", "call"}
                for region in (observation.get("regions") or {}).values() if region.get("kind") == "buttons"
                for item in region.get("items", []))
    return calls and not (observation.get("reaction_discard") or {}).get("confirmed")


def recapture_gaps(result):
    """How much a fresh frame could still fix: (incomplete regions, missing source)."""
    return len(incomplete_table_regions(result)) + int(reaction_source_missing(result))


def recapture_reasons(result):
    """Short Chinese names of what a fresh frame is expected to fix."""
    names = {"hand": "手牌", "draw": "摸牌", "dora": "宝牌指示牌"}
    reasons = []
    for name in incomplete_table_regions(result):
        kind = ((result.get("observation") or {}).get("regions") or {}).get(name, {}).get("kind")
        label = names.get(name) or {"river": "牌河", "meld": "副露"}.get(kind, name)
        if label not in reasons:
            reasons.append(label)
    if reaction_source_missing(result):
        reasons.append("出牌来源")
    return reasons or ["局面"]


def recapture_until_complete(frame, result, *, grab, process, timer_active, wait,
                             limit=MAX_RECAPTURES, delay=RECAPTURE_DELAY_SECONDS,
                             deadline=None, clock=time.perf_counter):
    """Take fresh frames while the board is incomplete and the turn is still ours.

    Retried gaps: incomplete board regions (animations, a window passing over)
    and call buttons without the pulsing last-discard outline. ``wait(delay)``
    returns True when stopping; ``timer_active()`` re-reads the countdown. The
    more complete result (ready, or fewer gaps) and its frame are kept.
    A deadline prevents starting a full retry that cannot fit the remaining
    budget, using the most recent actual analysis duration as its estimate.
    Returns ``(frame, result, recaptures)``; an incomplete result stays incomplete.
    """
    attempts = 0
    estimate=max(0,result.get('total_ms',0) or 0)/1000
    while result.get("status") == "needs_confirmation" and recapture_gaps(result) and attempts < limit:
        if deadline is not None and clock()+delay+estimate >= deadline:
            result['recapture_budget_exhausted']=True
            break
        if wait(delay) or not timer_active():
            break
        if deadline is not None and clock()+estimate >= deadline:
            result['recapture_budget_exhausted']=True
            break
        attempts += 1
        second = grab()
        started=clock()
        retry = process(second)
        estimate=max(0,clock()-started)
        if retry.get("status") == "ready" or (
                retry.get("observation") is not None and recapture_gaps(retry) < recapture_gaps(result)):
            frame, result = second, retry
    if attempts:
        result["recaptures"] = attempts
    return frame, result, attempts


def save_advice(path,response,pixels=None):
    path = Path(path)
    write_json(path,response)
    if pixels is not None:
        save_image(path.with_suffix(".overlay.png"),overlay(pixels,response["observation"]))


def format_advice(response):
    if response["status"] != "ready":
        return response.get("reason") or "等待局面确认。"
    rec = response["recommendation"]
    names = {"discard":"打出","riichi":"立直","chi":"吃","pon":"碰","ankan":"暗杠",
             "kakan":"加杠","daiminkan":"明杠","ron":"荣和","tsumo":"自摸",
             "pass":"跳过","kyushukyuhai":"九种九牌流局"}
    event = rec["event"]
    from app.ui.tile_text import tile_name
    text = names.get(rec["semantic"],rec["semantic"])
    if "pai" in event:
        text += " " + tile_name(event["pai"])
    if "consumed" in event:
        text += "（使用 " + "、".join(tile_name(t) for t in event["consumed"]) + "）"
    if rec["semantic"] == "riichi":
        followup = rec.get("after_riichi_declaration")
        text += ("，宣言后打出 "+tile_name(followup["event"]["pai"])) if followup else "；宣言后重新读取截图获取弃牌建议"
    return text
