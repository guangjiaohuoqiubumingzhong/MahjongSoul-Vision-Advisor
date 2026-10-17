from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from app.capture.frames import config_digest, load_image
from app.state.events import TILES
from .config import VisionError


def iou(left, right):
    x, y = max(left[0], right[0]), max(left[1], right[1])
    end_x, end_y = min(left[0]+left[2], right[0]+right[2]), min(left[1]+left[3], right[1]+right[3])
    intersection = max(0, end_x-x)*max(0, end_y-y)
    union = left[2]*left[3]+right[2]*right[3]-intersection
    return intersection/union if union else 0


def match_boxes(expected, actual, minimum_iou=.5):
    pairs = sorted([(iou(e["bbox"], a["bbox"]), ei, ai) for ei,e in enumerate(expected) for ai,a in enumerate(actual)], reverse=True)
    used_e, used_a, matches = set(), set(), []
    for score, ei, ai in pairs:
        if score < minimum_iou:
            break
        if ei not in used_e and ai not in used_a:
            used_e.add(ei)
            used_a.add(ai)
            matches.append((ei, ai))
    return matches, sorted(set(range(len(expected)))-used_e), sorted(set(range(len(actual)))-used_a)


def _metrics(count):
    return {**count, "detection_recall": count["matched"]/count["expected"] if count["expected"] else None,
            "classification_accuracy": count["correct"]/count["known_matched"] if count["known_matched"] else None,
            "end_to_end_accuracy": count["correct"]/count["expected"] if count["expected"] else None}


def evaluate(recognizer, paths: list[Path]) -> dict:
    loaded, splits, image_hashes = [], defaultdict(set), {}
    source_games = set(getattr(recognizer.bank,"training_games",recognizer.bank.manifest["source_games"]))
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if not annotation.get("reviewed") or not annotation.get("game_id"):
            raise VisionError(f"标注未经核对或没有对局编号：{path}")
        if annotation["split"] not in {"train", "debug", "test"}:
            raise VisionError("数据划分无效。")
        for target in annotation.get("targets", []):
            if target.get("kind") == "tile" and target.get("label") not in {*TILES, "?"}:
                raise VisionError(f"标注牌种无效：{target.get('label')} ({path})")
            if target.get("region") not in recognizer.config["regions"]:
                raise VisionError(f"标注区域无效：{target.get('region')} ({path})")
        expected_pause = annotation.get("expected_pause_regions", [])
        if not isinstance(expected_pause,list) or any(name not in recognizer.config["regions"] for name in expected_pause):
            raise VisionError(f"不确定画面的预期暂停区域无效：{path}")
        game = annotation["game_id"]
        splits[annotation["split"]].add(game)
        image_path = Path(annotation["image"])
        if not image_path.is_absolute():
            image_path = path.parent/image_path
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in image_hashes:
            raise VisionError(f"重复图片不能重复计入数据量：{path}、{image_hashes[digest]}")
        image_hashes[digest] = str(path)
        loaded.append((path, annotation, image_path))
    for split in ("train", "debug"):
        if splits[split] & splits["test"]:
            raise VisionError("同一对局不能跨调试/模板与验收集。")
    if source_games & (splits["debug"] | splits["test"]):
        raise VisionError("模板来源对局出现在 debug/test 集，验收无效。")
    # A second render of the same paused scene is useful training material, but
    # does not add a distinct position or new tile instances to the corpus gate.
    by_path={path.resolve():annotation for path,annotation,_ in loaded}
    def scene_labels(annotation):
        return Counter(json.dumps({key:target.get(key) for key in ("kind","region","label","value")},
                                  ensure_ascii=False,sort_keys=True) for target in annotation["targets"]
                       if target["kind"] in {"tile","text","button"})
    distinct=[]
    for path,annotation,image_path in loaded:
        flag=annotation.get("count_as_distinct_scene",True)
        if not isinstance(flag,bool):
            raise VisionError("不同局面计数标记必须是布尔值。")
        if flag:
            distinct.append((path,annotation,image_path))
            continue
        source=Path(annotation.get("duplicate_of",""))
        if not source.is_absolute():
            source=path.parent/source
        original=by_path.get(source.resolve())
        if (original is None or not original.get("count_as_distinct_scene",True) or
                original["game_id"]!=annotation["game_id"] or original["split"]!=annotation["split"] or
                scene_labels(original)!=scene_labels(annotation)):
            raise VisionError("重复局面需指向已核对的同对局/同划分且可见标签一致的原始标注。")
    test_frames, whole_correct, uncertain_cases, uncertain_paused = 0, 0, 0, 0
    counts = defaultdict(lambda: Counter(expected=0, matched=0, known_matched=0, correct=0, missed=0, false_positive=0, unknown=0))
    classes, rotations, tags = Counter(), set(), set()
    errors, results, text_total, text_correct, buttons_total, buttons_correct = [], [], 0, 0, 0, 0
    meld_total, meld_type_correct, meld_source_correct = 0, 0, 0
    for path, annotation, image_path in loaded:
        if annotation["split"] != "test" or not annotation.get("count_as_distinct_scene",True):
            continue
        observation = recognizer.recognize(load_image(image_path), {"image_path": str(image_path)})
        frame_ok = True
        tags.update(annotation.get("coverage_tags", []))
        if annotation.get("frame_quality") != "stable":
            uncertain_cases += 1
            expected_pause = set(annotation.get("expected_pause_regions", []))
            observed_pause = {issue.get("region") for issue in observation.get("issues", [])
                              if issue.get("reason") == "incomplete_region" and
                              issue.get("region") in observation["regions"] and
                              not observation["regions"][issue["region"]].get("complete",True)}
            verified = bool(expected_pause) and expected_pause <= observed_pause
            paused = observation["status"] == "needs_confirmation" and verified
            uncertain_paused += paused
            results.append({"annotation": str(path.resolve()), "quality": annotation.get("frame_quality"),
                            "status": observation["status"],"expected_pause_regions":sorted(expected_pause),
                            "observed_pause_regions":sorted(observed_pause),"pause_evidence_verified":verified})
            if not paused:
                errors.append({"annotation":str(path),"category":"uncertainty_pause_evidence_missing",
                               "expected_pause_regions":sorted(expected_pause),"observed_pause_regions":sorted(observed_pause),
                               "status":observation["status"]})
            continue
        test_frames += 1
        complete = set(annotation.get("complete_regions", []))
        required = {name for name,region in recognizer.config["regions"].items() if region.get("required")}
        if not required <= complete:
            frame_ok = False
            errors.append({"annotation": str(path), "category": "incomplete_ground_truth", "missing_regions": sorted(required-complete)})
        expected_by_region = defaultdict(list)
        for target in annotation["targets"]:
            if target["kind"] == "tile":
                expected_by_region[target["region"]].append(target)
                if target["label"] != "?":
                    classes[target["label"]] += 1
                    rotations.add(target.get("rotation", 0))
        for name in complete:
            region = observation["regions"][name]
            targets = [target for target in annotation["targets"] if target["region"] == name]
            if region["kind"] in {"hand", "draw", "river", "meld", "dora", "choices"}:
                expected, actual = expected_by_region[name], region["items"]
                matched, missing, extra = match_boxes(expected, actual)
                role = region["kind"]
                counts[role].update(expected=len(expected), matched=len(matched), missed=len(missing), false_positive=len(extra))
                for ei,ai in matched:
                    e, a = expected[ei], actual[ai]
                    correct = a["label"] == e["label"] and ("slot" not in e or e["slot"] == a.get("slot"))
                    counts[role]["known_matched"] += e["label"] != "?"
                    counts[role]["correct"] += correct and e["label"] != "?"
                    counts[role]["unknown"] += a["label"] == "?"
                    if e["label"].endswith("r"):
                        counts["red"].update(expected=1,matched=1,known_matched=1,correct=correct,unknown=a["label"] == "?")
                    if not correct:
                        frame_ok = False
                        errors.append({"annotation":str(path),"region":name,"category":"tile_classification_or_slot","expected":e,"actual":a})
                for index in missing:
                    if expected[index]["label"].endswith("r"):
                        counts["red"].update(expected=1,missed=1)
                if missing or extra:
                    frame_ok = False
                    errors.append({"annotation":str(path),"region":name,"category":"detection","missed":missing,"extra":extra})
                if region["kind"] == "meld":
                    expected_groups = defaultdict(list)
                    for item in expected:
                        if "group_id" in item:
                            expected_groups[item["group_id"]].append(item)
                    for items in expected_groups.values():
                        meld_total += 1
                        identities = {item["id"] for item in items}
                        actual_items = [actual[ai] for ei,ai in matched if expected[ei]["id"] in identities]
                        matched_instances = {id(item) for item in actual_items}
                        group = next((g for g in region.get("groups",[])
                                      if {id(item) for item in g["items"]} == matched_instances), None)
                        type_ok = group is not None and group["type"] == items[0].get("meld_type")
                        source_ok = group is not None and group["target"] == items[0].get("target") and group["target"] is not None
                        meld_type_correct += type_ok
                        meld_source_correct += source_ok
                        frame_ok = frame_ok and type_ok and source_ok
            elif region["kind"] == "text":
                expected = [item for item in targets if item["kind"] == "text"]
                if len(expected) != 1:
                    frame_ok = False
                    continue
                text_total += 1
                correct = region["text"]["value"] == expected[0].get("value") and expected[0].get("value") is not None
                text_correct += correct
                frame_ok = frame_ok and correct
                if not correct:
                    errors.append({"annotation":str(path),"region":name,"category":"text","expected":expected[0],"actual":region["text"]})
            elif region["kind"] == "buttons":
                expected = [item for item in targets if item["kind"] == "button"]
                matched, missing, extra = match_boxes(expected, region["items"])
                buttons_total += len(expected)
                correct = sum(expected[ei]["label"] == region["items"][ai]["label"] and expected[ei].get("state") == region["items"][ai]["state"] for ei,ai in matched)
                buttons_correct += correct
                frame_ok = frame_ok and correct == len(expected) and not missing and not extra
        frame_ok = frame_ok and observation["status"] == "recognized"
        whole_correct += frame_ok
        results.append({"annotation":str(path.resolve()),"whole_scene_correct":frame_ok,"recognition_ms":observation["recognition_ms"]})
    metrics = {role:_metrics(value) for role,value in counts.items()}
    failures = []
    total_tiles = sum(target["kind"] == "tile" and target["label"] != "?" for _,a,_ in loaded for target in a["targets"])
    distinct_tiles = sum(target["kind"] == "tile" and target["label"] != "?" for _,a,_ in distinct for target in a["targets"])
    if len(distinct) < 100 or distinct_tiles < 2000:
        failures.append("真实已核对数据尚未达到 100 张 / 2000 个可见牌实例")
    if not test_frames or len(splits["test"]) < 2:
        failures.append("缺少至少两场独立对局的稳定验收帧")
    if set(classes) != set(TILES):
        failures.append("37 种牌面独立验收样本覆盖不足")
    if rotations != {0,90,180,270}:
        failures.append("四方向验收覆盖不足")
    for role in ("hand", "draw", "river", "dora", "meld", "red"):
        m = metrics.get(role, {})
        if (m.get("detection_recall") or 0) < .995 or (m.get("classification_accuracy") or 0) < .995 or m.get("false_positive",0):
            failures.append(f"{role} 检测/分类目标未通过或缺样本")
    required_tags = {"chi","pon","daiminkan","ankan","kakan","stacked_kan","highlight","multiple_choices","animation","occlusion","low_resolution"}
    if not required_tags <= tags:
        failures.append("副露结构、高亮、组合选择或不确定画面专项覆盖不足")
    if not text_total or text_correct != text_total or not buttons_total or buttons_correct != buttons_total:
        failures.append("文字/操作按钮验收未通过或缺样本")
    if not meld_total or meld_type_correct != meld_total or meld_source_correct != meld_total:
        failures.append("副露类型与来源验收未通过或缺样本")
    if not uncertain_cases or uncertain_paused != uncertain_cases:
        failures.append("不确定画面暂停验证未通过或缺样本")
    if not recognizer.config.get("calibrated"):
        failures.append("客户区区域尚未完成校准")
    if test_frames and whole_correct == 0:
        failures.append("尚无完整稳定局面通过")
    return {"stage":2,"gate_passed":not failures,"status":"passed" if not failures else "incomplete",
            "config_version":recognizer.config["version"],"config_sha256":config_digest(recognizer.config),
            "template_manifest_sha256":getattr(recognizer.bank,"sha256",None),
            "classifier_manifest_sha256":getattr(recognizer.bank,"classifier_manifest_sha256",None),
            "feature_version":getattr(recognizer.bank,"feature_version",None),
            "tile_classifier":getattr(recognizer.bank,"tile_classifier_info",None),
            "evaluated_at":datetime.now(timezone.utc).isoformat(),"dataset_images":len(loaded),"dataset_known_tiles":total_tiles,
            "dataset_distinct_scenes":len(distinct),"dataset_distinct_scene_known_tiles":distinct_tiles,
            "dataset_visual_variants":len(loaded)-len(distinct),
            "dataset_frame_qualities":dict(Counter(a.get("frame_quality","unknown") for _,a,_ in loaded)),
            "splits":{k:sorted(v) for k,v in splits.items()},"stable_test_frames":test_frames,
            "test_class_coverage":dict(classes),"missing_classes":sorted(set(TILES)-set(classes)),"rotations":sorted(rotations),
            "regions":metrics,"text":{"correct":text_correct,"total":text_total},
            "buttons":{"correct":buttons_correct,"total":buttons_total},
            "melds":{"total":meld_total,"type_correct":meld_type_correct,"source_correct":meld_source_correct},
            "whole_scene_accuracy":whole_correct/test_frames if test_frames else None,
            "uncertain_frames":{"paused":uncertain_paused,"total":uncertain_cases},"gate_failures":failures,"errors":errors,"frames":results}
