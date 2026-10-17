"""Failed live retries remain usable as complete, separate reproduction cases."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.capture.frames import Frame, config_digest, load_image
from app.failure_archive import RecognitionFailureArchive


def reading(*missing, status="needs_confirmation"):
    return {"status": status, "reason": "请核对牌河。" if status == "needs_confirmation" else None,
            "recommendation": None, "observation": {"regions": {
                name: {"kind": kind, "complete": name not in missing,
                       "bbox": [0, 0, 4, 4], "items": []}
                for name, kind in (("hand", "hand"), ("river_right", "river"))}}}


def live_run(tmp_path, monkeypatch, rounds, *, cancel=None):
    import app.ui.advice as ui
    output = tmp_path / "screenshots"
    stop = threading.Event()
    state = SimpleNamespace(phase=0, counts=[0] * len(rounds), frames=[], published=[],
                            active=True, marker=0, closed=False, probes=0)
    config = {"version": "test-config", "client_size": [200, 120]}

    def wait(delay):
        # Finish each action's actual persistence before starting another one.
        saved_actions = len(list(output.glob("run-*/action-*/advice.json")))
        if delay == .15 and saved_actions > state.phase:
            if state.phase + 1 == len(rounds):
                stop.set()
            elif state.counts[state.phase] == len(rounds[state.phase]):
                state.phase += 1
                state.marker += 1
        return stop.is_set()
    stop.wait = wait

    def grab():
        frame = Frame(np.full((120, 200, 3), len(state.frames) + 1, np.uint8),
                      {"frame_id": len(state.frames) + 101, "captured_at": "original-time"}, config)
        state.frames.append(frame)
        return frame

    def process(pixels, metadata):
        index = state.counts[state.phase]
        state.counts[state.phase] += 1
        result = deepcopy(rounds[state.phase][index])
        if state.counts[state.phase] == 2:
            if cancel == "stop": stop.set()
            elif cancel == "ended": state.active = False
            elif cancel == "changed": state.marker += 1
        return result

    def probe(*args):
        state.probes += 1
        if state.probes > 60: stop.set()
        return state.active, 299, (13, True, state.marker)

    capture = SimpleNamespace(grab=grab, close=lambda: setattr(state, "closed", True))
    service = SimpleNamespace(clear=lambda: None, process=process)
    monkeypatch.setattr(ui, "WindowCapture", lambda *args: capture)
    monkeypatch.setattr(ui, "action_probe", probe)
    panel = ui.AdvicePanel.__new__(ui.AdvicePanel)
    panel.output, panel.config = output, config
    panel.get_service = lambda count: service
    panel.publish = lambda generation, result, pixels: state.published.append(deepcopy(result))
    panel.live_worker(1, stop, 123)
    return state, output, tmp_path / "recognition-errors"


def manifests(root):
    return list(root.glob("*/action-*/manifest.json"))


def test_live_failure_keeps_all_original_attempts_and_the_selected_earlier_frame(tmp_path, monkeypatch):
    readings = [reading("hand", "river_right"), reading("river_right"),
                reading("hand", "river_right"), reading("hand", "river_right")]
    state, output, errors = live_run(tmp_path, monkeypatch, [readings])
    assert state.closed and state.counts == [4]
    manifest_path, = manifests(errors)
    case = manifest_path.parent
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["recaptures"] == 3 and report["selected_attempt"] == 1
    assert report["review_status"] == "unreviewed" and report["split"] == "debug"
    assert len(report["attempts"]) == 4
    for index, entry in enumerate(report["attempts"]):
        image = case / entry["image"]
        assert np.array_equal(load_image(image), state.frames[index].pixels)
        assert entry["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        metadata = json.loads((case / entry["metadata"]).read_text(encoding="utf-8"))
        response = json.loads((case / entry["recognition"]).read_text(encoding="utf-8"))
        configuration = json.loads((case / entry["config"]).read_text(encoding="utf-8"))
        assert metadata["frame_id"] == index + 101 and metadata["recognition_attempt"] == index
        assert metadata["captured_at"] == "original-time"
        assert config_digest(configuration) == metadata["region_config_sha256"]
        assert Path(metadata["region_config_snapshot"]) == (case / entry["config"]).resolve()
        assert response["observation"]["frame"] == metadata
        assert response["observation"]["regions"] == readings[index]["observation"]["regions"]
    final = json.loads((case / report["final_result"]).read_text(encoding="utf-8"))
    assert final["recaptures"] == 3 and final["observation"]["frame"]["recognition_attempt"] == 1
    assert Path(json.loads((output / "latest.json").read_text(encoding="utf-8"))["failure_archive"]) == case.resolve()
    assert len(list(output.glob("run-*/action-*/[0-9]*.png"))) == 1


@pytest.mark.parametrize("cancel", ["stop", "ended", "changed"])
def test_stopping_or_changing_action_during_retries_does_not_archive_a_failure(tmp_path, monkeypatch, cancel):
    state, _, errors = live_run(tmp_path, monkeypatch, [[reading("river_right")] * 4], cancel=cancel)
    assert state.closed and not errors.exists()


def test_recovered_and_first_try_successes_do_not_create_failure_folders(tmp_path, monkeypatch):
    state, _, errors = live_run(tmp_path, monkeypatch, [
        [reading("river_right"), reading(status="ready")], [reading(status="ready")]])
    assert state.counts == [2, 1] and not errors.exists()


def test_two_failed_actions_share_a_run_and_preserve_separate_cases(tmp_path, monkeypatch):
    _, _, errors = live_run(tmp_path, monkeypatch, [[reading("river_right")] * 4] * 2)
    cases = manifests(errors)
    assert len(cases) == 2 and len({p.parent.parent for p in cases}) == 1
    assert {p.parent.name for p in cases} == {"action-0001", "action-0002"}
    assert len(list(errors.rglob("*.png"))) == 8


def test_budget_exhaustion_after_a_retry_preserves_the_frames_already_taken(tmp_path, monkeypatch):
    import app.ui.advice as ui
    original = ui.recapture_until_complete
    def limited(*args, **kwargs):
        clock = [0.0]
        process = kwargs["process"]
        def slower(frame):
            clock[0] = 1.0
            return process(frame)
        kwargs.update(process=slower, deadline=1.5, clock=lambda: clock[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(ui, "recapture_until_complete", limited)
    state, _, errors = live_run(tmp_path, monkeypatch, [[reading("river_right")] * 2])
    manifest_path, = manifests(errors)
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert state.counts == [2] and report["recapture_budget_exhausted"]
    assert report["recaptures"] == 1 and len(report["attempts"]) == 2


def test_no_archive_when_the_budget_prevents_even_the_first_retry(tmp_path, monkeypatch):
    first = reading("river_right"); first["total_ms"] = 100000
    state, _, errors = live_run(tmp_path, monkeypatch, [[first]])
    assert state.counts == [1] and not errors.exists()


def test_archive_write_failure_is_visible_and_later_actions_still_run(tmp_path, monkeypatch):
    def denied(*args, **kwargs): raise PermissionError("测试：目录不可写")
    monkeypatch.setattr(RecognitionFailureArchive, "save", denied)
    state, _, _ = live_run(tmp_path, monkeypatch, [
        [reading("river_right")] * 4, [reading(status="ready")]])
    assert state.counts == [4, 1] and state.closed
    assert any("目录不可写" in r.get("failure_archive_error", "") for r in state.published)
    assert any(r["status"] == "ready" for r in state.published)
    assert not any(r["status"] == "error" for r in state.published)


def test_same_named_runs_do_not_overwrite_and_original_frame_metadata_is_not_changed(tmp_path):
    config = {"version": "test"}
    frames = [Frame(np.full((10, 20, 3), i, np.uint8), {"frame_id": 1}, config) for i in (1, 2)]
    attempts = [(frame, reading("river_right")) for frame in frames]
    cases = [RecognitionFailureArchive(tmp_path, "same-run").save(
        attempts, frames[0], reading("river_right"), config=config, action_index=1, action_serial=1)
        for _ in range(2)]
    assert cases[0] != cases[1] and all((case / "manifest.json").exists() for case in cases)
    assert all(frame.metadata == {"frame_id": 1} for frame in frames)


def test_archive_rejects_a_changed_capture_configuration(tmp_path):
    config = {"version": "original"}
    frame = Frame(np.zeros((10, 20, 3), np.uint8),
                  {"frame_id": 1, "region_config_sha256": config_digest(config)}, {"version": "changed"})
    archive = RecognitionFailureArchive(tmp_path, "run")
    with pytest.raises(ValueError, match="配置发生变化"):
        archive.save([(frame, reading("river_right"))] * 2, frame, reading("river_right"),
                     config=config, action_index=1, action_serial=1)
    assert not list(tmp_path.rglob("manifest.json"))
