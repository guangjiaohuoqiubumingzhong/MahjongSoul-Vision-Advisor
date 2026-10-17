"""Keep every failed recognition attempt for later inspection and correction."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from app.advice import incomplete_table_regions
from app.capture.frames import config_digest, save_capture_config, save_image
from app.vision.config import write_json


class RecognitionFailureArchive:
    def __init__(self, output, run_id):
        # Separate program runs even if their start time and request number match.
        self.directory = Path(output) / f"{run_id}-{uuid4().hex[:12]}"

    def save(self, attempts, selected_frame, result, *, config, action_index, action_serial):
        """Write an unreviewed case only after at least one unsuccessful retry.

        The caller confirms that this is still the same action. Attempts contain
        the original Frame and a snapshot of its recognition result; successful
        and cancelled actions do not create an archive. The manifest is written
        last so an interrupted write cannot look like a complete case.
        """
        if len(attempts) < 2 or result.get("status") != "needs_confirmation":
            return None
        selected = next(index for index, (frame, _) in enumerate(attempts)
                        if frame is selected_frame)
        directory = self.directory / f"action-{action_index:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        entries, saved_metadata = [], []
        for index, (frame, reading) in enumerate(attempts):
            configuration = frame.configuration if frame.configuration is not None else config
            digest = config_digest(configuration)
            if frame.metadata.get("region_config_sha256", digest) != digest:
                raise ValueError("帧对应的采集配置发生变化，无法保存错误样本。")
            snapshot = save_capture_config(directory, configuration)
            stem = f"{index:02d}-" + ("initial" if index == 0 else f"retry-{index:02d}")
            image = directory / f"{stem}.png"
            metadata = deepcopy(frame.metadata)
            metadata.update(image_path=str(image.resolve()),
                            region_config_sha256=digest,
                            region_config_snapshot=str(snapshot.resolve()),
                            capture_trigger="automatic_action", action_serial=action_serial,
                            recognition_attempt=index)
            save_image(image, frame.pixels)
            write_json(image.with_suffix(".frame.json"), metadata)
            saved_metadata.append(metadata)
            response = deepcopy(reading)
            if response.get("observation") is not None:
                response["observation"]["frame"] = metadata
            recognition = image.with_suffix(".recognition.json")
            write_json(recognition, response)
            entries.append({"attempt": index, "kind": "initial" if index == 0 else "retry",
                            "image": image.name, "metadata": image.with_suffix(".frame.json").name,
                            "recognition": recognition.name,
                            "config": snapshot.name,
                            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                            "status": reading.get("status"), "reason": reading.get("reason"),
                            "incomplete_regions": incomplete_table_regions(reading),
                            "selected": index == selected})
        final = deepcopy(result)
        if final.get("observation") is not None:
            final["observation"]["frame"] = saved_metadata[selected]
        write_json(directory / "final-result.json", final)
        write_json(directory / "manifest.json", {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "trigger": "automatic_recaptures_failed", "review_status": "unreviewed", "split": "debug",
            "run_id": self.directory.name, "action_index": action_index, "action_serial": action_serial,
            "config_version": config.get("version"), "status": result["status"],
            "reason": result.get("reason"), "recaptures": len(attempts) - 1,
            "recapture_budget_exhausted": bool(result.get("recapture_budget_exhausted")),
            "selected_attempt": selected, "final_result": "final-result.json", "attempts": entries})
        return directory
