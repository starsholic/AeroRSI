"""From verified repairs to the weighted supervision of the next policy.

* ``pool``: the accepted anchors of a round become repair rows (input, prompt, verified action).
* ``relabel``: the current policy's own greedy outputs on the normal-state inputs; these are the retention targets,
  so that learning the repairs does not move the policy elsewhere.
* ``assemble``: weighted best-of-group SFT rows. Repairs carry 80% of the total weight and normal states 20%. From
  round 2 on, the previous round's training repairs are replayed with 30% of the repair weight. 20% of the repair
  trajectory groups are held out of training altogether (their normal states too).
"""
from collections import Counter
from pathlib import Path

from ..actions import mode, native_format
from ..utils import digest, file_hash, read_json, read_jsonl, write_jsonl

# Hash salts of the paper's runs, kept so that this code reproduces their held-out groups and row order exactly.
SALT_HELDOUT, SALT_ORDER = "e066-repair-heldout", "e066-train-order"


def pool(collection, output):
    """repairs.jsonl: one row per accepted anchor of a collection run."""
    collection = Path(collection)
    plan = read_json(collection / "plan.json")
    results = [read_json(p) for p in sorted(collection.glob("tasks/*/*/result.json"))]
    planned, found = {r["id"] for r in plan["records"]}, {r["route_id"] for r in results}
    if not found <= planned:
        raise ValueError("The collection contains routes outside its plan")
    status, rows = Counter(), []
    for result in results:
        for state in result["states"]:
            status[state["status"]] += 1
            if state["status"] != "accepted":
                continue
            rows.append({"id": digest([plan["identity"], result["route_id"], state["step"], state["chosen"]]),
                         "route_id": result["route_id"], "group": result["group"], "scene": result["scene"],
                         "step": state["step"], "offset": state["offset"], "image": state["image"],
                         "image_sha256": state["image_sha256"], "prompt": state["prompt"], "chosen": state["chosen"],
                         "rejected": state["rejected"], "chosen_name": state["chosen_name"],
                         "chosen_progress_m": state["chosen_progress_m"],
                         "chosen_confirmation_progress_m": state["chosen_confirmation_progress_m"],
                         "manifest_identity": plan["manifest_identity"]})
    rows.sort(key=lambda r: r["id"])
    sha = write_jsonl(Path(output) / "repairs.jsonl", rows)
    return {"planned_routes": len(planned), "completed_routes": len(found), "missing_routes": len(planned - found),
            "anchor_status": dict(status), "accepted": len(rows), "accepted_groups": len({r["group"] for r in rows}),
            "chosen_sources": dict(Counter(r["chosen_name"] for r in rows)),
            "chosen_modes": dict(Counter(mode(r["chosen"]) for r in rows)), "repairs_sha256": sha}


def relabel(config, adapter, inputs, output):
    """predictions.jsonl: the policy's greedy answer on every normal-state input (one GPU)."""
    from PIL import Image
    from ..policy import Policy
    model = config["model"]
    policy = Policy(model["base"], adapter, model["attention"])
    rows, predictions = read_jsonl(inputs), []
    for row in rows:
        if file_hash(row["image"]) != row["image_sha256"]:
            raise ValueError(f"Input image changed: {row['image']}")
        text = policy.generate(policy.inputs(Image.open(row["image"]).convert("RGB"), row["prompt"]))
        predictions.append({"id": row["id"], "text": text, "mode": mode(text)})
    sha = write_jsonl(Path(output) / "predictions.jsonl", predictions)
    return {"inputs": len(rows), "modes": dict(Counter(p["mode"] for p in predictions)), "predictions_sha256": sha}


def heldout_groups(repairs, seed, fraction):
    groups = sorted({r["group"] for r in repairs}, key=lambda g: digest([seed, SALT_HELDOUT, g]))
    return set(groups[:round(len(groups) * fraction)])


def assemble(repairs_dir, plan_dir, predictions, output, seed, s, replay_dir=None):
    """train_rows.jsonl (weighted supervision, in training order) and heldout_repairs.jsonl."""
    repairs = read_jsonl(Path(repairs_dir) / "repairs.jsonl")
    normal = read_jsonl(Path(plan_dir) / "normal_train.jsonl")
    if len({r["id"] for r in repairs}) != len(repairs):
        raise ValueError("Repair rows overlap")
    held = heldout_groups(repairs, seed, s["heldout_fraction"])
    train = [r for r in repairs if r["group"] not in held]
    heldout = [r for r in repairs if r["group"] in held]
    # Normal states share trajectory groups with repairs; held-out groups stay entirely unseen.
    dropped = sum(r["group"] in held for r in normal)
    normal = [r for r in normal if r["group"] not in held]
    predicted = {r["id"]: r["text"] for r in read_jsonl(predictions)}
    if not {r["id"] for r in normal} <= set(predicted):
        raise ValueError("The current policy's predictions must cover every normal input")
    relabeled = [dict(r, target_text=predicted[r["id"]]) for r in normal if native_format(predicted[r["id"]])]
    nonnative, normal = len(normal) - len(relabeled), relabeled
    replay = []
    if replay_dir is not None:
        replay = [r for r in read_jsonl(Path(replay_dir) / "train_rows.jsonl") if r["role"] == "repair" and r["group"] not in held]
    if not train or not normal:
        raise ValueError("Need both repair and normal rows")
    # Weights: every repair row the same, every normal row the same, the shares fixed; the mean weight is 1.
    share, replay_share = s["repair_weight"], s["replay_fraction"] if replay else 0.0
    total = len(train) + len(replay) + len(normal)
    w_repair = share * total * (1 - replay_share) / len(train)
    w_normal = (1 - share) * total / len(normal)
    rows = [{"id": r["id"], "group": r["group"], "role": "repair", "image": r["image"], "image_sha256": r["image_sha256"],
             "prompt": r["prompt"], "target_text": r["chosen"], "supervision_weight": w_repair,
             "manifest_identity": r["manifest_identity"]} for r in train]
    rows += [{"id": r["id"], "group": r["group"], "role": "normal", "image": r["image"], "image_sha256": r["image_sha256"],
              "prompt": r["prompt"], "target_text": r["target_text"], "supervision_weight": w_normal,
              "manifest_identity": r["manifest_identity"]} for r in normal]
    if replay:  # replayed rows keep their relative weights
        scale = share * total * replay_share / sum(r["supervision_weight"] for r in replay)
        rows += [dict(r, role="replay", supervision_weight=r["supervision_weight"] * scale) for r in replay]
    if any(not native_format(r["target_text"]) for r in rows):
        raise ValueError("Every target must use the policy-native command format")
    if len({r["id"] for r in rows}) != len(rows) or {r["group"] for r in heldout} & {r["group"] for r in rows}:
        raise ValueError("Duplicate rows, or a held-out group leaked into training")
    rows.sort(key=lambda r: digest([seed, SALT_ORDER, r["id"]]))
    output = Path(output)
    sha = write_jsonl(output / "train_rows.jsonl", rows)
    write_jsonl(output / "heldout_repairs.jsonl", heldout)
    return {"train_repairs": len(train), "heldout_repairs": len(heldout), "heldout_groups": len(held),
            "normal_rows": len(normal), "normal_rows_in_heldout_groups": dropped, "normal_nonnative_dropped": nonnative,
            "replay_rows": len(replay), "repair_weight": w_repair, "normal_weight": w_normal,
            "repair_sources": dict(Counter(r["chosen_name"] for r in train)),
            "repair_target_modes": dict(Counter(mode(r["target_text"]) for r in rows if r["role"] == "repair")),
            "train_rows_sha256": sha}
