"""Normal-state inputs: where the next policy is trained to keep the current policy's own actions.

They are taken once, from the round-0 policy's own successful flights on the training routes: per flight, the state
before its first moving action, the state closest to 30 m from the target (between 20 and 40 m), and the state before
its final stop. Each round relabels these inputs with its own policy (``assemble.relabel``), so their targets always
come from the policy being trained.
"""
from collections import Counter
from pathlib import Path

from ..utils import digest, read_json, write_jsonl

SALT = "normal-flights"


def select_states(trace):
    """[(step, roles)] of the normal states of a flight; empty unless it ended with a successful stop."""
    if not trace or not trace[-1]["valid"] or not trace[-1]["stop"] or trace[-1]["terminal"] != "success":
        return []
    moving = [r for r in trace if r["valid"] and not r["stop"]]
    approach = [r for r in moving if 20 < r["distance_before_m"] <= 40]
    choices = [("early", moving[0] if moving else None),
               ("approach", min(approach, key=lambda r: (abs(r["distance_before_m"] - 30), r["step"])) if approach else None),
               ("successful_stop", trace[-1])]
    selected = {}
    for role, row in choices:
        if row is not None:
            selected.setdefault(row["step"], []).append(role)
    return sorted(selected.items())


def collect(rollouts, manifest_identity, seed, flights, output):
    """normal_inputs.jsonl from up to `flights` successful flights, one per trajectory group, in a seeded order."""
    successful = {}
    for run in rollouts:
        for path in sorted(Path(run).glob("tasks/*/*/result.json")):
            result = read_json(path)
            if result.get("status") == "complete" and result["success"] and result.get("normal_states"):
                successful.setdefault(result["group"], result)  # one flight per trajectory group
    groups = sorted(successful, key=lambda g: digest([seed, SALT, g]))[:flights]
    rows = [{"id": digest(["normal", successful[g]["route_id"], s["step"]]), "group": g, "route_id": successful[g]["route_id"],
             "step": s["step"], "roles": s["roles"], "image": s["image"], "image_sha256": s["image_sha256"],
             "prompt": s["prompt"], "manifest_identity": manifest_identity}
            for g in groups for s in successful[g]["normal_states"]]
    sha = write_jsonl(Path(output) / "normal_inputs.jsonl", rows)
    return {"successful_groups": len(successful), "flights": len(groups), "inputs": len(rows),
            "roles": dict(Counter(role for r in rows for role in r["roles"])), "normal_inputs_sha256": sha}
