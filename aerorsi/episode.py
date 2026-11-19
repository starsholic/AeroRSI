"""Observation geometry, episode termination and per-route metrics.

Termination follows the public TravelUAV/AeroDPO evaluator: an episode ends at a depth collision (more than 10% of
any depth camera's pixels within 1 m), when the drone has been stuck (horizontal motion below 1 m for more than 25
consecutive actions), when it has moved away from the target for ten consecutive actions, at a stop within 20 m
of the target (success), or after the action limit. A stop farther away is an early end that continues the episode
until another terminal condition, again as in the public evaluator.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

DEPTH_CAMERAS = ("frontcamera_depth", "leftcamera_depth", "rightcamera_depth", "rearcamera_depth", "downcamera_depth")
SUCCESS_RADIUS_M = 20.0
EVENTS = ("collision", "stuck", "wrong_way", "premature_stop")


def position(observation):
    return np.asarray(observation["sensors"]["state"]["position"], dtype=float)


def distance(observation, target) -> float:
    return float(np.linalg.norm(position(observation) - target))


def depth_collision(observation) -> str:
    """Name of the first depth camera that reports a collision, or ''."""
    for name, depth in zip(DEPTH_CAMERAS, observation["depth"], strict=True):
        if np.count_nonzero(depth <= 1) > 0.1 * depth.size:
            return name
    return ""


def clearance(observation, camera: int = 0) -> float:
    """10th-percentile depth of one camera in metres (depth is stored as uint8 over 0..100 m)."""
    return float(np.percentile(observation["depth"][camera], 10) / 255 * 100)


def bearing(observation, target) -> float:
    """Horizontal angle of the target in the body frame (radians, positive to the right)."""
    q = observation["sensors"]["state"]["orientation"]
    q = [q.get(k, 1 if k == "w" else 0) for k in ("x", "y", "z", "w")] if isinstance(q, dict) else q
    vector = Rotation.from_quat(q).inv().apply(np.asarray(target) - position(observation))
    return float(np.arctan2(vector[1], vector[0]))


@dataclass
class EpisodeStatus:
    target: list
    distances: list = field(default_factory=list)
    last_position: object = None
    stationary: int = 0
    success: bool = False
    oracle_success: bool = False
    early_end: bool = False
    done: bool = False
    terminal: str = ""
    counts: dict = field(default_factory=lambda: dict.fromkeys(EVENTS, 0))

    def start(self, observation):
        self.distances = [distance(observation, self.target)]

    def update(self, observation, prediction, *, physical_collision=False):
        """Advance by one executed action; returns the depth-collision camera ('' if none).

        The first post-action position counts as one stationary step and equal distances count as moving away,
        as in the public evaluator. Physical contact without a depth collision is recorded but does not end the
        episode, so the collision rate keeps its official definition.
        """
        d = distance(observation, self.target)
        p = position(observation)
        previous = p if self.last_position is None else self.last_position
        self.stationary = self.stationary + 1 if np.linalg.norm(p[:2] - previous[:2]) < 1 else 0
        self.last_position = p
        self.distances.append(d)
        self.oracle_success |= d < SUCCESS_RADIUS_M
        source = depth_collision(observation)
        stuck = self.stationary > 25
        wrong_way = len(self.distances) >= 10 and all(b >= a for a, b in zip(self.distances[-10:-1], self.distances[-9:]))
        events = {"collision": bool(source) or physical_collision, "stuck": stuck, "wrong_way": wrong_way,
                  "premature_stop": prediction.stop and d > SUCCESS_RADIUS_M}
        for key, value in events.items():
            self.counts[key] += int(value)
        if source or stuck or wrong_way:
            self.done = True
            self.terminal = "collision" if source else "stuck" if stuck else "wrong_way"
        elif prediction.stop:
            if d <= SUCCESS_RADIUS_M and not self.early_end:
                self.success = True
            elif d > SUCCESS_RADIUS_M:
                self.early_end = True
            self.done = self.success or (self.oracle_success and self.early_end)
            if self.done:
                self.terminal = "success" if self.success else "early_end"
        return source

    def to_dict(self):
        row = asdict(self)
        if row["last_position"] is not None:
            row["last_position"] = np.asarray(row["last_position"]).tolist()
        return row


def path_length(points) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def route_metrics(route, positions, status: EpisodeStatus, steps: int) -> dict:
    """Per-route SR/OSR/SPL/NE inputs, computed as in the official metric script (utils/metric_aerodpo.py)."""
    reference = [p["position"] for p in route["trajectory"]]
    reference_length = path_length(reference)
    flown_length = path_length(positions)
    shortened = reference_length - SUCCESS_RADIUS_M  # the official SPL subtracts the success radius
    denominator = max(shortened, flown_length)
    spl = max(shortened / denominator, 0.0) if status.success and denominator > 0 else 0.0
    return {"route_id": route["id"], "group": route["group"], "split": route["split"], "category": route["category"],
            "terminal": status.terminal or "timeout", "success": status.success,
            "oracle_success": status.oracle_success or status.success, "collision": status.terminal == "collision",
            "steps": steps, "spl": spl, "ne_m": float(np.linalg.norm(np.asarray(reference[-1]) - np.asarray(positions[-1]))),
            "reference_length_m": reference_length, "flown_path_length_m": flown_length,
            "final_target_distance_m": status.distances[-1], "event_counts": status.counts}
