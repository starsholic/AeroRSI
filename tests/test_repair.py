import json

import pytest

from aerorsi.repair.anchors import probe_horizon
from aerorsi.repair.assemble import assemble
from aerorsi.repair.verify import failure_reproduced, qualifies, score

S = {"collision_penalty": 100.0, "stop_penalty": 100.0, "success_bonus": 20.0, "min_progress_m": 4.0,
     "advantage_margin_m": 4.0, "horizon": 8, "horizon_extra": 4, "horizon_max": 20}


def branch(terminal, progress, valid=True):
    b = {"valid": valid, "terminal": terminal, "progress_m": progress}
    b["score"] = score(b, S)
    return b


def test_score_penalties_and_bonus():
    assert branch("horizon", 6.0)["score"] == 6.0
    assert branch("collision", 6.0)["score"] == -94.0
    assert branch("premature_stop", 1.0)["score"] == -99.0
    assert branch("success", 2.0)["score"] == 22.0
    assert score({"valid": False}, S) is None


def test_qualifies():
    assert qualifies(branch("success", 0.5), S)  # a successful stop always qualifies
    assert qualifies(branch("horizon", 4.0), S)
    assert not qualifies(branch("horizon", 3.9), S)
    assert not qualifies(branch("collision", 10.0), S)


def test_failure_reproduced():
    assert failure_reproduced(branch("collision", 5.0), [], S)
    best = [branch("horizon", 10.0)]
    assert failure_reproduced(branch("horizon", 5.9), best, S)
    assert not failure_reproduced(branch("horizon", 6.1), best, S)


def test_probe_horizon_reaches_past_the_failure():
    assert [probe_horizon(k, S) for k in (1, 2, 4, 8, 16)] == [8, 8, 8, 12, 20]


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_assemble_weights_and_heldout(tmp_path):
    repairs = [{"id": f"r{i}", "group": f"g{i}", "image": "x.png", "image_sha256": "0", "prompt": "p",
                "chosen": "52 49 49", "chosen_name": "sample", "manifest_identity": "m"} for i in range(10)]
    normal = [{"id": f"n{i}", "group": f"h{i}", "image": "y.png", "image_sha256": "0", "prompt": "p",
               "target_text": "40 49 49", "manifest_identity": "m"} for i in range(30)]
    (tmp_path / "pool").mkdir(), (tmp_path / "plan").mkdir(), (tmp_path / "out").mkdir()
    write(tmp_path / "pool/repairs.jsonl", repairs)
    write(tmp_path / "plan/normal_train.jsonl", normal)
    write(tmp_path / "pred.jsonl", [{"id": r["id"], "text": "45 49 49"} for r in normal])
    a = {"repair_weight": 0.8, "replay_fraction": 0.3, "heldout_fraction": 0.2}
    m = assemble(tmp_path / "pool", tmp_path / "plan", tmp_path / "pred.jsonl", tmp_path / "out", 1, a)
    rows = [json.loads(line) for line in (tmp_path / "out/train_rows.jsonl").read_text().splitlines()]
    assert m["heldout_repairs"] == 2 and m["train_repairs"] == 8 and m["normal_rows"] == 30
    total = sum(r["supervision_weight"] for r in rows)
    assert total == pytest.approx(len(rows))  # mean weight 1
    repair = sum(r["supervision_weight"] for r in rows if r["role"] == "repair")
    assert repair / total == pytest.approx(0.8)
    assert all(r["target_text"] == "45 49 49" for r in rows if r["role"] == "normal")  # current-policy relabels
