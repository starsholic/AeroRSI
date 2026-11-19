"""Continuation-verified repairs: the policy judges its own candidate actions by flying on from each of them.

At an anchor the simulator is restored to the paused state before the failed action. The candidates are the failed
action itself, the policy's own samples and the rule-based proposals. Each candidate is executed and then followed by
the *current* policy for a short horizon; its score is the progress toward the target, with a penalty for crashing or
stopping early and a bonus for reaching the target. The anchor yields a repair when

1. the failure reproduces: the failed action fails again, or the best qualifying candidate beats it by a margin;
2. some candidate qualifies: it reaches the target, or survives the horizon with enough progress; and
3. the best qualifying candidate qualifies again when flown a second time (confirmation; one more candidate may try).

A candidate is thus judged by where the policy can take it, not by how it looks one step ahead.
"""
import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..actions import Prediction, decode, mode, native_format
from ..data import load_route
from ..episode import SUCCESS_RADIUS_M, depth_collision, distance, position
from ..policy import Policy, prompt
from ..sim.world import World
from ..utils import digest, file_hash, write_json
from .proposals import PotentialField, propose

BAD_TERMINALS = ("collision", "premature_stop", "invalid_policy_output")


def score(branch, s):
    if not branch["valid"]:
        return None
    value = branch["progress_m"]
    if branch["terminal"] in ("collision", "invalid_policy_output"):
        value -= s["collision_penalty"]
    elif branch["terminal"] == "premature_stop":
        value -= s["stop_penalty"]
    elif branch["terminal"] == "success":
        value += s["success_bonus"]
    return value


def qualifies(branch, s):
    """A successful stop always qualifies; otherwise the candidate must survive the horizon with enough progress."""
    if branch["valid"] and branch["terminal"] == "success":
        return True
    return branch["valid"] and branch["terminal"] in ("horizon", "success") and branch["progress_m"] >= s["min_progress_m"]


def failure_reproduced(original, eligible, s):
    if original["terminal"] in BAD_TERMINALS:
        return True
    return bool(eligible) and eligible[0]["score"] - original["score"] >= s["advantage_margin_m"]


def sample_allowed(text, s):
    """Samples enter the group only in the native format; a native stop only if stop candidates are allowed."""
    return native_format(text) and mode(text) != "invalid" and (mode(text) != "LAND" or s["stop_candidates"])


def saved_observation(anchor):
    """The anchor's saved policy input: front/down RGB (BGR order, as observations hold) and all depth images."""
    image = np.asarray(Image.open(anchor["image"]).convert("RGB"))
    half = image.shape[0] // 2
    rgb = [np.zeros_like(image[:half]) for _ in range(5)]
    rgb[0], rgb[4] = image[:half, ..., ::-1].copy(), image[half:, ..., ::-1].copy()
    with np.load(anchor["depth"]) as data:
        depth = [x.copy() for x in data["depth"]]
    return {"rgb": rgb, "depth": depth, "sensors": anchor["before_sensors"], "physical_collision": False}


class RepairWorker:
    def __init__(self, slot, output, config, adapter):
        self.slot, self.output, self.config, self.adapter = slot, Path(output), config, adapter
        self.s = config["repair"]

    def __enter__(self):
        import torch
        torch.set_num_threads(self.config["resources"]["cpu_threads_per_worker"])
        model = self.config["model"]
        self.policy = Policy(model["base"], self.adapter, model["attention"])
        self.field = PotentialField()
        self.world = World(self.slot, self.output, self.config["paths"]["envs"]).__enter__()
        return self

    def __exit__(self, *args):
        self.world.__exit__(*args)

    def continuation(self, route, snapshot, first_text, horizon):
        """Execute first_text from the restored snapshot, then let the current policy fly up to horizon actions."""
        if not self.world.restore(snapshot):
            return {"valid": False, "reason": "restore_mismatch", "restore_errors": self.world.last_restore_errors}
        restore_errors = self.world.last_restore_errors
        target = route["target"]
        start = np.asarray(snapshot["position"], dtype=float)
        d0 = float(np.linalg.norm(start - np.asarray(target)))
        action, stop, valid = decode(first_text)
        prediction = Prediction(action, stop, first_text, valid)
        steps, terminal = [], "horizon"
        for index in range(horizon):
            observation = self.world.step(prediction.action, paused=True)
            source, physical = depth_collision(observation), bool(self.world.last_physical_collision)
            d = distance(observation, target)
            steps.append({"text": prediction.text, "distance_m": d, "collision_source": source,
                          "physical_collision": physical, "xy": position(observation)[:2].tolist()})
            if source or physical:
                terminal = "collision"
                break
            if prediction.stop:
                terminal = "success" if d <= SUCCESS_RADIUS_M else "premature_stop"
                break
            if index == horizon - 1:
                break
            prediction = self.policy.predict(observation, target, route["instruction"])
            if not prediction.valid:
                terminal = "invalid_policy_output"
                steps.append({"text": prediction.text, "invalid": True})
                break
        flown = [r for r in steps if "xy" in r]
        path = sum(float(np.linalg.norm(np.asarray(b["xy"]) - np.asarray(a["xy"])))
                   for a, b in zip([{"xy": start[:2].tolist()}] + flown, flown))
        return {"valid": True, "terminal": terminal, "actions": len(flown), "progress_m": d0 - flown[-1]["distance_m"],
                "xy_path_m": path, "steps": steps, "restore_errors": restore_errors}

    def verify(self, route, anchor):
        s = self.s
        if file_hash(anchor["image"]) != anchor["image_sha256"]:
            raise ValueError(f"Saved policy input changed: {anchor['image']}")
        text = prompt(anchor["snapshot"], route["target"], route["instruction"])
        inputs = self.policy.inputs(Image.open(anchor["image"]).convert("RGB"), text)
        greedy = self.policy.generate(inputs)
        record = {"step": anchor["step"], "offset": anchor["offset"], "rejected": anchor["rejected"], "prompt": text,
                  "image": anchor["image"], "image_sha256": anchor["image_sha256"], "greedy": greedy}
        if greedy != anchor["rejected"]:  # the saved input must reproduce the recorded action
            return {**record, "status": "input_mismatch"}
        seed = int(digest([self.config["seed"], route["id"], anchor["step"]])[:8], 16)
        record["sampled"] = self.policy.samples(inputs, s["policy_samples"], seed, s["temperature"])
        candidates, seen = [("original", anchor["rejected"])], {anchor["rejected"]}
        for sample in record["sampled"]:
            if sample_allowed(sample, s) and sample not in seen:
                candidates.append(("sample", sample))
                seen.add(sample)
        for name, proposal in propose(self.field, saved_observation(anchor), route["target"], anchor["rejected_action"],
                                      anchor["terminal_collision_source"], s["proposals"]):
            if proposal not in seen and mode(proposal) != "LAND":
                candidates.append((name, proposal))
                seen.add(proposal)
        horizon = anchor["horizon"]
        record.update(failure=anchor["failure"], horizon=horizon)
        branches = []
        for name, candidate in candidates:
            branch = self.continuation(route, anchor["snapshot"], candidate, horizon)
            branches.append({"name": name, "text": candidate, "mode": mode(candidate), **branch, "score": score(branch, s)})
        record["branches"] = branches
        original = branches[0]
        if not original["valid"]:
            return {**record, "status": "negative_unreliable"}
        eligible = sorted((b for b in branches[1:] if b["valid"] and qualifies(b, s)),
                          key=lambda b: (-b["score"], b["name"] != "sample", b["text"]))
        if not failure_reproduced(original, eligible, s):
            return {**record, "status": "negative_not_reproduced"}
        if not eligible:
            return {**record, "status": "no_qualifying_candidate"}
        for branch in eligible[:s["confirm_attempts"]]:
            check = self.continuation(route, anchor["snapshot"], branch["text"], horizon)
            branch["confirmation"] = {**check, "score": score(check, s)}
            if check["valid"] and qualifies(check, s):
                return {**record, "status": "accepted", "chosen": branch["text"], "chosen_name": branch["name"],
                        "chosen_mode": branch["mode"], "chosen_score": branch["score"],
                        "chosen_confirmation_score": branch["confirmation"]["score"],
                        "chosen_progress_m": branch["progress_m"], "chosen_confirmation_progress_m": check["progress_m"],
                        "rejected_score": original["score"]}
        return {**record, "status": "confirmation_failed"}

    def run(self, record):
        route = load_route(record["route"])
        folder = self.output / digest(route["id"])[:20]
        folder.mkdir()
        write_json(folder / "route.json", record["route"])
        started = time.monotonic()
        try:
            self.world.reset(route)
            write_json(folder / "initial_state.json", self.world.last_reset_diagnostics)
            states = [self.verify(route, copy.deepcopy(anchor)) for anchor in record["states"]]
            write_json(folder / "result.json", {"route_id": route["id"], "group": route["group"], "scene": route["scene"],
                                                "split": "train", "source": record["source"], "status": "complete",
                                                "states": states, "elapsed_seconds": time.monotonic() - started})
        except BaseException as error:
            write_json(folder / "infrastructure_error.json", {"route_id": route["id"], "error": repr(error),
                                                              "elapsed_seconds": time.monotonic() - started})
            raise
