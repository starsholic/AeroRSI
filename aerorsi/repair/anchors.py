"""Anchors: states shortly before the current policy's failures on training routes.

Every failed rollout (collision, stuck, wrong way, early end or timeout) contributes probes 1, 2, 4, 8 and 16 actions
before its end. A fixed number of probes per round is then drawn uniformly over all probes, so that a failure mode
that has been repaired becomes rarer and automatically receives fewer anchors. Each anchor keeps the paused state
before its action, the action the policy took there, and the policy input it saw.

The normal-state inputs used for retention are split here as well: 20% of their trajectory groups are held out, and
no anchor is taken on a held-out group, so those states stay unseen by training.
"""
from collections import Counter
from pathlib import Path

from ..actions import native_format
from ..utils import digest, file_hash, read_json, read_jsonl, seeded_order, write_json, write_jsonl

FAILURE_TERMINALS = ("collision", "stuck", "wrong_way", "early_end", "timeout")
# Hash salts of the paper's runs, kept so that this code reproduces their anchors and splits exactly.
SALT_BUDGET, SALT_NORMAL = "probe-budget", "e066-normal-split"


def probe_horizon(offset, s):
    """Continuation length at an anchor: long enough to pass the original failure point."""
    return min(s["horizon_max"], max(s["horizon"], offset + s["horizon_extra"]))


def failure_probes(folder: Path, s):
    result = read_json(folder / "result.json")
    if result.get("status") != "complete" or result.get("split") != "train" or result.get("terminal") not in FAILURE_TERMINALS:
        return []
    trace = read_json(folder / "action_trace.json")
    probes = []
    for offset in s["probe_offsets"]:
        if len(trace) < offset:
            continue
        row = trace[-offset]
        if row["step"] <= s["min_step"] or not row["valid"] or not native_format(row["text"]):
            continue
        image = folder / "observations" / f"step_{row['step'] - 1:03d}.png"  # the input the policy acted on
        depth = image.with_suffix(".npz")
        if not image.exists() or not depth.exists():
            continue
        probes.append({"offset": offset, "step": row["step"], "rejected": row["text"], "rejected_action": row["action"],
                       "snapshot": row["before"]["state"], "before_sensors": row["before"], "image": str(image),
                       "image_sha256": file_hash(image), "depth": str(depth), "distance_before_m": row["distance_before_m"],
                       "failure": result["terminal"], "horizon": probe_horizon(offset, s),
                       "terminal_collision_source": trace[-1].get("collision_source") or ""})
    return probes


def split_normal(path, seed, fraction):
    """(train rows, held-out rows, held-out groups); relative image paths are resolved against the file's folder."""
    folder = Path(path).resolve().parent
    rows = [dict(r, image=str(folder / r["image"])) if not Path(r["image"]).is_absolute() else r for r in read_jsonl(path)]
    groups = seeded_order({r["group"] for r in rows}, seed, SALT_NORMAL)
    held = set(groups[:round(len(groups) * fraction)])
    return [r for r in rows if r["group"] not in held], [r for r in rows if r["group"] in held], held


def plan(config, manifest, rollouts, normal_inputs, output):
    """Write plan.json (the anchors, grouped by route) and the normal-input split of this round."""
    s, seed = config["repair"], config["seed"]
    train = {r["id"]: r for r in manifest["routes"] if r["split"] == "train"}
    normal_train, normal_eval, held = split_normal(normal_inputs, seed, config["normal"]["eval_fraction"])
    records, seen = [], set()
    skipped = Counter()
    for source in rollouts:
        for path in sorted(Path(source).glob("tasks/*/*/result.json")):
            saved = read_json(path.parent / "route.json")
            route = train.get(saved["id"])
            if route is None or route["group"] in held:
                skipped["not_train" if route is None else "held_out_normal_group"] += 1
                continue
            probes = failure_probes(path.parent, s)
            if not probes:
                continue
            if route["group"] in seen:  # one route per trajectory group
                skipped["duplicate_group"] += 1
                continue
            seen.add(route["group"])
            records.append({"id": route["id"], "scene": route["scene"], "group": route["group"], "route": route,
                            "source": str(path.parent), "states": probes})
    # The per-round budget, drawn uniformly over all probes.
    probes = sorted(((r["id"], st["step"]) for r in records for st in r["states"]),
                    key=lambda x: digest([seed, SALT_BUDGET, x[0], x[1]]))
    keep = set(probes[:s["anchor_budget"]])
    for r in records:
        r["states"] = [st for st in r["states"] if (r["id"], st["step"]) in keep]
    records = sorted((r for r in records if r["states"]), key=lambda r: (r["scene"], r["id"]))
    output = Path(output)
    hashes = {"normal_train": write_jsonl(output / "normal_train.jsonl", normal_train),
              "normal_eval": write_jsonl(output / "normal_eval.jsonl", normal_eval)}
    value = {"manifest_identity": manifest["identity"], "settings": s, "records": records}
    value["identity"] = digest(value)
    write_json(output / "plan.json", value)
    return {"plan_identity": value["identity"], "routes": len(records), "anchors": sum(len(r["states"]) for r in records),
            "anchors_by_offset": dict(Counter(st["offset"] for r in records for st in r["states"])),
            "anchors_by_failure": dict(Counter(st["failure"] for r in records for st in r["states"])),
            "skipped": dict(skipped), "normal_train_inputs": len(normal_train), "normal_eval_inputs": len(normal_eval),
            "normal_held_out_groups": len(held), "sha256": hashes}
