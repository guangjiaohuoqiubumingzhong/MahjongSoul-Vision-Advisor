"""Mortal decisions on fresh visible snapshots, independent of event sessions."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
from libriichi.state import PlayerState

from app.state.snapshot import validate_snapshot, digest, SnapshotError
from app.engine.actions import action_variants, semantic

ASSUMPTIONS = ["same_cycle_furiten_ignored_by_user_request", "unknown_tedashi_has_no_positive_feature",
               "call_timing_and_missing_called_river_slots_not_reconstructed",
               "ippatsu_and_rinshan_history_not_reconstructed", "fixed_four_player_red_five_hanchan_model"]


class SnapshotBackend:
    def __init__(self,runtime):
        self.runtime = runtime
        from app.backend import Backend
        self.rules = Backend(runtime).rules

    def legal_ids(self,snapshot):
        """Rule-mask indices the position allows, or ``None`` if it is invalid.

        Cheap probe used to settle which visible meld was just completed after
        a call: the rules engine already knows whether a position is legal
        (食替 and the other discard constraints included), so no inference runs.
        """
        try:
            validate_snapshot(snapshot)
            state = PlayerState.from_snapshot(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
            _,mask = state.encode_obs(3,False)
        except (SnapshotError,ValueError,TypeError,KeyError):
            return None
        ids = [int(i) for i in np.flatnonzero(mask)]
        return ids or None

    def decide(self,snapshot,allowed_actions=None):
        started = time.perf_counter()
        validate_snapshot(snapshot)
        if not hasattr(PlayerState,"from_snapshot"):
            raise SnapshotError("后台扩展缺少当前局面接口，请执行 scripts/build.ps1。")
        state = PlayerState.from_snapshot(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
        hand = Counter(snapshot["hand"])
        if snapshot["draw"] is not None:
            hand[snapshot["draw"]] += 1
        ledger = SimpleNamespace(player_id=snapshot["player_id"],hand=hand,melds=snapshot["melds"])
        obs, rule_mask = state.encode_obs(3,False)
        mask = rule_mask.copy()
        variants = {}
        for i in np.flatnonzero(rule_mask):
            i = int(i)
            actions = action_variants(i,state,ledger)
            for action in actions:
                state.validate_reaction(json.dumps(action))
            if not actions:
                raise SnapshotError("合法动作无法转换为具体操作。")
            variants[i] = actions
            if allowed_actions is not None:
                kinds = {semantic(a) for a in actions}
                aliases = {"riichi":"reach","daiminkan":"kan","ankan":"kan","kakan":"kan"}
                mask[i] = any(k in allowed_actions or aliases.get(k) in allowed_actions for k in kinds)
        if not mask.any():
            raise SnapshotError("当前按钮与规则合法动作没有交集。")
        with self.runtime.lock:
            q,elapsed = self.runtime.run(np.asarray([obs]),np.asarray([mask]))
            candidates = [{"action_id":int(i),"semantic":semantic(variants[int(i)][0]),
                           "model_score":float(q[0,i]),"variants":deepcopy(variants[int(i)])}
                          for i in np.flatnonzero(mask)]
            candidates.sort(key=lambda a:(-a["model_score"],a["action_id"]))
            selected = candidates[0]
            action = deepcopy(selected["variants"][0])
            if selected["action_id"] == 42 and len(selected["variants"]) > 1:
                kan_obs,kan_mask = state.encode_obs(3,True)
                kan_q,kan_elapsed = self.runtime.run(np.asarray([kan_obs]),np.asarray([kan_mask]))
                elapsed += kan_elapsed
                tile_id = int(kan_q[0].argmax())
                from app.state.events import TILES,base
                action = next(v for v in selected["variants"] if base(v.get("pai",v["consumed"][0])) == base(TILES[tile_id]))
                selected["selection_scores"] = [{"tile":TILES[int(i)],"model_score":float(kan_q[0,i])}
                                                 for i in np.flatnonzero(kan_mask)]
        state.validate_reaction(json.dumps(action))
        before = list(state.tehai)
        response = {"status":"ready", "state_kind":"current_screenshot_snapshot",
                    "snapshot_sha256":digest(snapshot),"assumptions":ASSUMPTIONS,
                    "recommendation":{**{k:v for k,v in selected.items() if k != "variants"},"event":action},
                    "legal_candidates":candidates,"shanten":state.shanten,
                    "model_version":self.runtime.model_version,"execution_provider":"CUDAExecutionProvider",
                    "rules_id":self.rules["id"],
                    "score_label":"模型评分（Q 值）","inference_ms":elapsed,
                    "total_ms":(time.perf_counter()-started)*1000,
                    "observation_sha256":hashlib.sha256(obs.tobytes()).hexdigest(),
                    "rule_mask":rule_mask.tolist(),"action_mask":mask.tolist()}
        if selected["semantic"] == "riichi":
            followup = deepcopy(snapshot)
            declared = list(followup.get("riichi_declared",followup["riichi"]))
            declared[followup["player_id"]] = True
            followup["riichi_declared"] = declared
            response["recommendation"]["after_riichi_declaration"] = self.decide(followup,["discard"])["recommendation"]
            response["total_ms"] = (time.perf_counter()-started)*1000
        assert before == list(state.tehai), "建议不能执行到当前局面"
        return response
