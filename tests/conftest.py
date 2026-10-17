import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def runtime():
    from app.engine.runtime import OnnxRuntime
    return OnnxRuntime()


@pytest.fixture
def feed():
    def receive(session, event, **overrides):
        return session.receive({"session_id": session.session_id, "generation": session.generation,
                                "event_seq": session.event_seq + 1, "state_version": session.state_version,
                                "event": event, **overrides})
    return receive


def load_sample(name):
    return [json.loads(line) for line in (ROOT / "data/fixtures" / f"{name}.mjai.jsonl").read_text(encoding="utf-8").splitlines()]
