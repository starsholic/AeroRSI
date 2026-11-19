"""Fly whole routes with the current policy: evaluation, and the rollouts that expose its failures.

Evaluation steps the simulator without pausing, exactly as the public evaluator. Rollouts on training routes pause
after every action (the collection cadence), record each action with the full state before it, and keep the policy
inputs of the last 17 steps of every failed episode: the anchors of a round lie at most 16 actions before a failure.
Successful rollouts keep the inputs of their three normal states instead (see ``repair/normal.py``).
"""
from collections import deque
import time
from pathlib import Path

import numpy as np

from .data import load_route
from .episode import EpisodeStatus, distance, position, route_metrics
from .policy import Policy, mosaic, prompt
from .repair.normal import select_states
from .sim.world import World
from .utils import digest, file_hash, write_json

KEEP_LAST = 17  # the deepest anchor is 16 actions before the failure; its input is the observation one step earlier


def save_observation(observation, path):
    """Policy input (mosaic PNG) and all five depth images (NPZ) of one observation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mosaic(observation).save(path.with_suffix(".png"))
    np.savez_compressed(path.with_suffix(".npz"), depth=np.stack(observation["depth"]))


class FlightWorker:
    def __init__(self, slot, output, config, mode, adapter):
        if mode not in ("rollout", "evaluate"):
            raise ValueError(mode)
        self.slot, self.output, self.config, self.mode, self.adapter = slot, Path(output), config, mode, adapter

    def __enter__(self):
        import torch
        torch.set_num_threads(self.config["resources"]["cpu_threads_per_worker"])
        torch.manual_seed(self.config["seed"])
        model = self.config["model"]
        self.policy = Policy(model["base"], self.adapter, model["attention"])
        self.world = World(self.slot, self.output, self.config["paths"]["envs"]).__enter__()
        return self

    def __exit__(self, *args):
        self.world.__exit__(*args)

    def run(self, record):
        route = load_route(record)
        folder = self.output / digest(route["id"])[:20]
        folder.mkdir()
        write_json(folder / "route.json", record)
        started = time.monotonic()
        paused = self.mode == "rollout"
        try:
            observation = self.world.reset(route)
            write_json(folder / "initial_state.json", self.world.last_reset_diagnostics)
            status = EpisodeStatus(route["target"])
            status.start(observation)
            positions, trace = [position(observation).tolist()], []
            recent = deque([("step_000", observation)], maxlen=KEEP_LAST + 1)
            candidates = {}  # step -> observation before it, for the states that select_states may pick
            step = 0
            for step in range(1, self.config["protocol"]["max_actions"] + 1):
                before = observation
                prediction = self.policy.predict(before, route["target"], route["instruction"])
                observation = self.world.step(prediction.action, paused=paused)
                source = status.update(observation, prediction, physical_collision=self.world.last_physical_collision)
                positions.append(position(observation).tolist())
                trace.append({"step": step, "before": before["sensors"], "after": observation["sensors"],
                              "text": prediction.text, "action": prediction.action, "valid": prediction.valid,
                              "stop": prediction.stop, "distance_before_m": distance(before, route["target"]),
                              "distance_after_m": distance(observation, route["target"]), "collision_source": source,
                              "physical_collision": self.world.last_physical_collision, "terminal": status.terminal})
                recent.append((f"step_{step:03d}", observation))
                moving = prediction.valid and not prediction.stop
                if paused and (moving and not candidates or moving and 20 < trace[-1]["distance_before_m"] <= 40
                               or status.done):
                    candidates[step] = before
                if status.done:
                    break
            if paused and not status.success:
                for name, saved in recent:
                    save_observation(saved, folder / "observations" / name)
            normal_states = []
            if paused and status.success:  # normal-state inputs from a successful flight (see repair/normal.py)
                for selected, roles in select_states(trace):
                    image = folder / "normal" / f"before_action_{selected:03d}.png"
                    image.parent.mkdir(exist_ok=True)
                    mosaic(candidates[selected]).save(image)
                    normal_states.append({"step": selected, "roles": roles, "image": str(image.resolve()),
                                          "image_sha256": file_hash(image),
                                          "prompt": prompt(candidates[selected]["sensors"]["state"], route["target"],
                                                           route["instruction"])})
            write_json(folder / "action_trace.json", trace)
            result = route_metrics(route, positions, status, step)
            write_json(folder / "result.json", {**result, "mode": self.mode, "status": "complete",
                                                "normal_states": normal_states,
                                                "elapsed_seconds": time.monotonic() - started})
        except BaseException as error:
            write_json(folder / "infrastructure_error.json", {"route_id": route["id"], "error": repr(error),
                                                              "elapsed_seconds": time.monotonic() - started})
            raise
