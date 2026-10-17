from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
import time

import numpy as np
from libriichi.mjai import Bot
from libriichi.state import PlayerState

from app.engine.actions import action_variants, semantic
from app.engine.adapter import MortalOnnxAdapter
from app.engine.runtime import ROOT
from app.state.events import ACTION_EVENTS, TILES, EventLedger, InputError, UnsupportedRule, visible_event


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Session:
    """One player's actual state. Recommendations are never fed back into it."""
    def __init__(self, session_id, player_id, runtime, rules_id="mortal-four-player-aka-hanchan-v1"):
        if not isinstance(session_id, str) or not session_id or type(player_id) is not int or player_id not in range(4):
            raise InputError("需要非空会话 ID 和 0–3 玩家编号。")
        self.session_id, self.player_id = session_id, player_id
        self.runtime, self.rules_id = runtime, rules_id
        self.lock = threading.RLock()
        self.generation = -1
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.adapter = MortalOnnxAdapter(self.runtime)
            self.bot = Bot(self.adapter, self.player_id)
            self.state = PlayerState(self.player_id)
            self.ledger = EventLedger(self.player_id)
            self.event_seq = self.state_version = 0
            self.history, self.requests = [], {}
            self.sync_status, self.reason = "synchronized", None
            self.last_response = None
            return self._response("waiting_action")

    def _response(self, status, **values):
        return {"session_id": self.session_id, "generation": self.generation,
                "event_seq": self.event_seq, "state_version": self.state_version,
                "round_id": self.ledger.round_id, "status": status,
                "sync_status": self.sync_status, "reason": self.reason,
                "recommendation": None, "legal_candidates": [],
                "model_version": self.runtime.model_version, "rules_id": self.rules_id,
                "gpu_id": self.runtime.device_id, "execution_provider": "CUDAExecutionProvider",
                "score_label": "模型评分（Q 值）", **values}

    def _stop(self, status, reason):
        self.sync_status, self.reason = "unsynchronized", reason
        self.last_response = None
        return self._response(status)

    def _rebuild(self):
        self.state = PlayerState(self.player_id)
        self.adapter = MortalOnnxAdapter(self.runtime)
        self.bot = Bot(self.adapter, self.player_id)
        for event in self.history:
            line = canonical(event)
            self.state.update(line)
            self.bot.react(line, can_act=False)

    def receive(self, request: dict):
        with self.lock, self.runtime.lock:
            started = time.perf_counter()
            if self.sync_status != "synchronized":
                return self._response("unsynchronized")
            try:
                if request.get("session_id") != self.session_id:
                    raise InputError("会话 ID 不匹配。")
                if type(request.get("generation")) is not int or request["generation"] != self.generation:
                    raise InputError("reset generation 不匹配；旧对局事件不能进入新状态。")
                seq, version = request.get("event_seq"), request.get("state_version")
                if type(seq) is not int or seq < 1 or type(version) is not int:
                    raise InputError("需要连续正整数 event_seq 和输入前的 state_version。")
                event = visible_event(request.get("event"), self.player_id)
                fingerprint = canonical({"event": event, "state_version": version})
                if seq in self.requests:
                    if self.requests[seq] != fingerprint:
                        raise InputError("重复序号的事件内容或状态版本发生冲突。")
                    return self._response("duplicate", duplicate_event_seq=seq)
                if seq != self.event_seq + 1:
                    return self._stop("unsynchronized", f"事件缺失/乱序：预期 {self.event_seq + 1}，收到 {seq}。")
                if version != self.state_version:
                    return self._stop("unsynchronized", "输入状态版本与当前状态不一致。")
                pending = deepcopy(self.ledger)
                pending.accept(event)
                if event.get("actor") == self.player_id and event["type"] in ACTION_EVENTS:
                    self.state.validate_reaction(canonical(event))
                    if event["type"] == "dahai":
                        _, previous_mask = self.state.encode_obs(3, False)
                        if not previous_mask[TILES.index(event["pai"])]:
                            raise InputError("实际弃牌被规则掩码禁止（例如食替/立直约束）。")
                if event["type"] == "ryukyoku" and "deltas" not in event and self.state.last_cans.can_ryukyoku:
                    self.state.validate_reaction(canonical(event))
                line = canonical(event)
                self.state.update(line)
                if event["type"] not in ("start_game", "end_game", "end_kyoku", "hora", "ryukyoku"):
                    expected_count = 13 - 3 * len(pending.melds[self.player_id]) + (pending.pending_discard == self.player_id)
                    if sum(pending.hand.values()) != expected_count or sum(self.state.tehai) != expected_count:
                        raise InputError("自己的手牌数量与行动阶段/副露数不一致。")
                self.adapter.last_batch = None
                reaction_json = self.bot.react(line)
                self.ledger = pending
                self.event_seq, self.state_version = seq, version + 1
                self.history.append(event)
                self.requests[seq] = fingerprint
                if reaction_json is None:
                    status = "ended" if self.ledger.ended else "waiting_action"
                    self.last_response = self._response(status, total_ms=(time.perf_counter() - started) * 1000)
                else:
                    self.last_response = self._decision(json.loads(reaction_json), started)
                return deepcopy(self.last_response)
            except UnsupportedRule as exc:
                return self._stop("rule_unsupported", str(exc))
            except (InputError, KeyError, TypeError, ValueError) as exc:
                self._rebuild()
                return self._stop("input_incomplete", str(exc))
            except BaseException as exc:
                # PyO3 panic exceptions derive from BaseException; never continue with partial state.
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self._rebuild()
                return self._stop("error", f"规则/推理失败，恢复最后确认历史并暂停：{exc}")

    def _decision(self, reaction, started):
        batch = self.adapter.last_batch
        if batch is None or not self.state.last_cans.can_act:
            raise RuntimeError("没有匹配当前状态的模型推理。")
        obs, mask = self.state.encode_obs(3, False)
        if not np.array_equal(mask, batch["masks"][-1]):
            raise RuntimeError("Bot 和当前状态的合法掩码不一致。")
        observation_hash = hashlib.sha256(obs.tobytes()).hexdigest()
        if observation_hash != batch["observation_sha256"][-1]:
            raise RuntimeError("Bot 与可见状态的特征不一致。")
        main_q = batch["q"][-1]
        candidates = []
        for index in np.flatnonzero(mask):
            index = int(index)
            variants = action_variants(index, self.state, self.ledger)
            if not variants:
                raise RuntimeError(f"合法模型动作 {index} 缺少具体操作。")
            for variant in variants:
                self.state.validate_reaction(canonical(variant))
            entry = {"action_id": index, "semantic": semantic(variants[0]),
                     "model_score": float(main_q[index]), "variants": variants}
            if index == 37:
                entry["next_step"] = "等待确认立直宣言事件，再输出合法宣言弃牌"
            if index == 42 and len(batch["q"]) > 1:
                # Separate Q values refer to choosing the kan tile, not its top-level desirability.
                entry["selection_scores"] = [
                    {"tile_action_id": int(i), "tile": TILES[int(i)],
                     "selection_score": float(batch["q"][0][i])}
                    for i in np.flatnonzero(batch["masks"][0])]
                entry["selection_scores"].sort(key=lambda value: -value["selection_score"])
            candidates.append(entry)
        candidates.sort(key=lambda value: (-value["model_score"], value["action_id"]))
        action = {key: value for key, value in reaction.items() if key != "meta"}
        self.state.validate_reaction(canonical(action))
        selected_id = batch["actions"][-1]
        if selected_id != candidates[0]["action_id"] or not mask[selected_id]:
            raise RuntimeError("推荐不是合法掩码中的最高评分动作。")
        selected = next(item for item in candidates if item["action_id"] == selected_id)
        if canonical(action) not in {canonical(v) for v in selected["variants"]}:
            # consumed ordering has no semantic effect; normalize it for comparison.
            normalized = deepcopy(action)
            if "consumed" in normalized:
                normalized["consumed"].sort()
            options = []
            for variant in selected["variants"]:
                variant = deepcopy(variant)
                if "consumed" in variant:
                    variant["consumed"].sort()
                options.append(canonical(variant))
            if canonical(normalized) not in options:
                raise RuntimeError("Mortal 的具体反应不在统一动作候选中。")
        return self._response("ready", recommendation={"action_id": selected_id, "semantic": semantic(action),
                                                      "event": action, "model_score": selected["model_score"]},
                              legal_candidates=candidates, inference_ms=batch["inference_ms"],
                              total_ms=(time.perf_counter() - started) * 1000,
                              observation_sha256=observation_hash,
                              is_greedy=True, shanten=self.state.shanten,
                              upstream_meta=reaction.get("meta"))

    def decide(self, state_version: int):
        with self.lock:
            if state_version != self.state_version:
                return self._response("stale_request")
            if self.sync_status != "synchronized":
                return self._response("unsynchronized")
            return deepcopy(self.last_response or self._response("waiting_action"))


class Backend:
    """Multiple isolated sessions share one GPU model; session IDs cannot be overwritten."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.rules = json.loads((ROOT / "configs/rules.json").read_text(encoding="utf-8"))
        if (self.rules["id"] != "mortal-four-player-aka-hanchan-v1" or self.rules["players"] != 4
                or self.rules["red_fives"] != [1, 1, 1] or self.rules["match"] != "hanchan"):
            raise UnsupportedRule("规则配置未通过当前适配范围验收。")
        self.sessions = {}
        self.lock = threading.RLock()

    def initialize(self, session_id, player_id):
        with self.lock:
            if session_id in self.sessions:
                raise InputError("会话已存在，请显式 reset 或使用新 ID。")
            session = Session(session_id, player_id, self.runtime, self.rules["id"])
            self.sessions[session_id] = session
            return session
