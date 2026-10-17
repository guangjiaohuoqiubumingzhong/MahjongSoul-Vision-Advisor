from __future__ import annotations

import hashlib
import numpy as np

from .runtime import OnnxRuntime


class MortalOnnxAdapter:
    """The exact four-list react_batch ABI used by the pinned MortalBatchAgent."""
    engine_type = "mortal"
    is_oracle = False
    version = 3
    enable_quick_eval = False
    enable_rule_based_agari_guard = False
    name = "majan-onnx-cuda-fp32"

    def __init__(self, runtime: OnnxRuntime):
        self.runtime = runtime
        self.last_batch = None

    def react_batch(self, obs, masks, invisible_obs):
        if invisible_obs is not None:
            raise ValueError("禁止将对手暗手或 oracle 观察传入模型。")
        observations = np.stack(obs)
        legal = np.stack(masks)
        q, elapsed = self.runtime.run(observations, legal)
        actions = q.argmax(axis=1)
        self.last_batch = {"actions": actions.tolist(), "q": q, "masks": legal,
                           "inference_ms": elapsed,
                           "observation_sha256": [hashlib.sha256(row.tobytes()).hexdigest() for row in observations]}
        return actions.tolist(), q.tolist(), legal.tolist(), [True] * len(actions)
