from copy import deepcopy
import json

import numpy as np
import pytest

from app.backend import Backend
from app.engine.adapter import MortalOnnxAdapter
from app.engine.runtime import EnvironmentError, read_profile
from conftest import ROOT, load_sample

SAMPLES = json.loads((ROOT / "data/fixtures/manifest.json").read_text(encoding="utf-8")) if (ROOT / "data/fixtures/manifest.json").exists() else []


@pytest.mark.gpu
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample["name"])
def test_fixed_scenes(runtime, feed, sample):
    session = Backend(runtime).initialize(sample["name"], sample["player_id"])
    expectations = {e["event_seq"]: e for e in sample["expectations"]}
    events = [json.loads(line) for line in (ROOT / "data/fixtures" / sample["file"]).read_text(encoding="utf-8").splitlines()]
    for seq, event in enumerate(events, 1):
        response = feed(session, event)
        assert response["sync_status"] == "synchronized", response
        assert response["state_version"] == seq
        if seq in expectations:
            assert response["status"] == "ready", response
            expected = expectations[seq]
            candidates = response["legal_candidates"]
            legal = {candidate["action_id"] for candidate in candidates}
            assert set(expected.get("legal_ids", [])) <= legal
            assert not set(expected.get("illegal_ids", [])) & legal
            if "exact_legal_ids" in expected:
                assert legal == set(expected["exact_legal_ids"])
            assert set(expected.get("semantics", [])) <= {candidate["semantic"] for candidate in candidates}
            if "kan_choice_ids" in expected:
                kan = next(c for c in candidates if c["action_id"] == 42)
                assert {s["tile_action_id"] for s in kan["selection_scores"]} == set(expected["kan_choice_ids"])
        if response["status"] == "ready":
            before = list(session.state.tehai)
            for candidate in response["legal_candidates"]:
                for action in candidate["variants"]:
                    session.state.validate_reaction(json.dumps(action))
            session.state.validate_reaction(json.dumps(response["recommendation"]["event"]))
            assert before == list(session.state.tehai), "建议不能执行到真实状态"


@pytest.mark.gpu
def test_dedup_gap_and_reset(runtime, feed):
    session = Backend(runtime).initialize("duplicate", 0)
    events = load_sample("discard_red")
    for event in events:
        assert feed(session, event)["sync_status"] == "synchronized"
    before = list(session.state.tehai)
    duplicate = {"session_id": session.session_id, "generation": 0,
                 "event_seq": 3, "state_version": 2, "event": events[-1]}
    assert session.receive(duplicate)["status"] == "duplicate"
    assert session.state_version == 3 and before == list(session.state.tehai)
    assert session.receive({**duplicate, "event_seq": 5, "state_version": 3})["status"] == "unsynchronized"
    assert session.decide(3)["recommendation"] is None
    session.reset()
    assert session.generation == 1
    assert feed(session, events[0])["status"] == "waiting_action"
    assert session.receive(duplicate)["status"] == "input_incomplete"


@pytest.mark.gpu
def test_end_reset_and_session_isolation(runtime, feed):
    backend = Backend(runtime)
    one, two = backend.initialize("one", 0), backend.initialize("two", 0)
    for event in load_sample("tsumo"):
        response = feed(one, event)
        assert response["sync_status"] == "synchronized", response
    assert response["status"] == "ended"
    assert two.state_version == 0 and not two.history
    one.reset()
    for event in load_sample("discard_red"):
        response = feed(one, event)
    assert response["status"] == "ready"
    assert one.runtime.session is two.runtime.session
    assert one.decide(0)["status"] == "stale_request"


@pytest.mark.gpu
def test_hidden_information_cannot_change_features(runtime, feed):
    events = load_sample("chi_red_choices")
    full = deepcopy(events)
    full[1]["tehais"][1:] = [["C"] * 13, ["1m"] * 13, ["5sr"] * 13]
    full[2]["pai"] = "N"
    backend = Backend(runtime)
    responses = []
    for name, log in (("public", events), ("omniscient", full)):
        session = backend.initialize(name, 0)
        for event in log:
            response = feed(session, event)
        assert response["status"] == "ready", response
        assert all(hand == ["?"] * 13 for hand in session.history[1]["tehais"][1:])
        assert session.history[2]["pai"] == "?"
        responses.append(response)
    assert responses[0]["observation_sha256"] == responses[1]["observation_sha256"]
    assert responses[0]["legal_candidates"] == responses[1]["legal_candidates"]
    assert responses[0]["recommendation"] == responses[1]["recommendation"]


@pytest.mark.gpu
@pytest.mark.parametrize("mutation", ["missing_tsumogiri", "unknown_self", "bad_count", "wrong_actor", "unsupported_west", "wrong_version"])
def test_bad_input_pauses(runtime, feed, mutation):
    events = deepcopy(load_sample("discard_red"))
    session = Backend(runtime).initialize(mutation, 0)
    if mutation == "bad_count":
        events[1]["tehais"][0][:5] = ["E"] * 5
    elif mutation == "unknown_self":
        events[1]["tehais"][0][0] = "?"
    elif mutation == "unsupported_west":
        events[1]["bakaze"] = "W"
    elif mutation == "wrong_actor":
        events[2]["actor"] = 2
    for event in events:
        overrides = {"state_version": 8} if mutation == "wrong_version" else {}
        response = feed(session, event, **overrides)
        if response["sync_status"] != "synchronized":
            break
    if mutation == "missing_tsumogiri":
        response = feed(session, {"type": "dahai", "actor": 0, "pai": "N"})
    assert response["sync_status"] == "unsynchronized", response
    assert response["recommendation"] is None


def test_profiler_rejects_cpu_core_and_records_shape_nodes(tmp_path):
    path = tmp_path / "profile.json"
    core = {"cat": "Node", "name": "conv_kernel_time", "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}}
    path.write_text(json.dumps([core]))
    with pytest.raises(EnvironmentError):
        read_profile(path)
    core["args"]["provider"] = "CUDAExecutionProvider"
    shape = {"cat": "Node", "name": "shape_kernel_time", "args": {"op_name": "Shape", "provider": "CPUExecutionProvider"}}
    path.write_text(json.dumps([core, shape]))
    result = read_profile(path)
    assert result["core_all_cuda"] and result["non_cuda_nodes"] == {"Shape:shape_kernel_time": 1}


@pytest.mark.gpu
def test_oracle_and_shape_rejected(runtime):
    adapter = MortalOnnxAdapter(runtime)
    with pytest.raises(ValueError, match="oracle"):
        adapter.react_batch([], [], [])
    with pytest.raises(ValueError):
        runtime.run(np.zeros((1, 938, 34), dtype=np.float32), np.ones((1, 46), dtype=np.bool_))


@pytest.mark.gpu
def test_cuda_unavailable_is_startup_failure(runtime, monkeypatch):
    import onnxruntime as ort
    from app.engine.runtime import OnnxRuntime
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(EnvironmentError, match="不可用"):
        OnnxRuntime()
