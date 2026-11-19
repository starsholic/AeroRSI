"""Rule-based repair proposals, which join the policy's own samples as candidates at an anchor.

``apf`` is the collision-direction-filtered artificial potential field action of AeroDPO: it replaces only the axis
that the collision camera implicates (altitude for the down camera, yaw for a side camera, the more deviating of the
two for the front camera) and keeps the failed action elsewhere. ``apf_translate`` keeps the failed forward motion
and clips the APF yaw below the turn threshold, so the drone still translates. ``climb`` raises the failed action
by 2 m. No proposal is trusted: each one is verified like any sampled action.
"""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from ..actions import decode, encode


class PotentialField:
    """AeroDPO's APF expert: target attraction plus repulsion from the front and down depth images."""
    safe_distance, safe_down = 12.0, 5.0
    k_att, k_rep, k_rep_down = 1.0, 10.0, 10.0
    max_fwd, max_yaw, max_down = 5.0, 1.1, 5.0
    fov = np.radians(90.0)

    def __init__(self):
        self.theta_h = self.theta_v = None

    def action(self, observation, target, rejected, collision_source=""):
        chosen = copy.deepcopy(rejected)
        state = observation["sensors"]["state"]
        q = state["orientation"]
        quat = [q.get("x", 0), q.get("y", 0), q.get("z", 0), q.get("w", 1)] if isinstance(q, dict) else list(q)
        world = np.asarray(target) - np.asarray(state["position"])
        distance = np.linalg.norm(world)
        body = Rotation.from_quat(quat).inv().apply(world)
        attraction = body / distance * self.k_att if distance > 1e-3 else np.zeros(3)

        front = observation["depth"][0].astype(np.float32) / 255.0 * 100.0
        if self.theta_h is None:
            h, w = front.shape
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            self.theta_h, self.theta_v = (x - w / 2) / w * self.fov, (y - h / 2) / h * self.fov
        near = front < self.safe_distance
        repulsion = np.zeros(3)
        if np.any(near):
            magnitude = self.k_rep * (self.safe_distance - np.clip(front[near], 0.1, self.safe_distance)) / self.safe_distance
            cos_v, sin_v = np.cos(self.theta_v[near]), np.sin(self.theta_v[near])
            repulsion = np.array([np.sum(-magnitude * cos_v * np.cos(self.theta_h[near])),
                                  np.sum(-magnitude * cos_v * np.sin(self.theta_h[near])),
                                  np.sum(-magnitude * sin_v)]) / np.sum(near)
        down = observation["depth"][4].astype(np.float32) / 255.0 * 100.0
        h, w = down.shape
        lowest = max(np.min(down[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]), 0.1)
        lift = (np.array([0.0, 0.0, -self.k_rep_down * (self.safe_down - lowest) / self.safe_down])
                if lowest < self.safe_down else np.zeros(3))
        total = attraction + repulsion + lift

        apf_yaw = np.clip(np.arctan2(total[1], max(total[0], 0.5)), -self.max_yaw, self.max_yaw)
        apf_down = np.clip(total[2], -self.max_down, self.max_down)
        speed = np.clip((np.min(front) - 1.0) / 3.0, 0.1, 1.0) if np.any(near) else 1.0
        apf_fwd = np.clip(np.linalg.norm(total[:2]) * speed, 0.0, self.max_fwd)

        if collision_source == "downcamera_depth":
            chosen["down"] = apf_down
        elif collision_source in ("leftcamera_depth", "rightcamera_depth"):
            chosen["yaw"] = apf_yaw
        elif collision_source == "frontcamera_depth":
            if abs(apf_down - chosen["down"]) / self.max_down > abs(apf_yaw - chosen["yaw"]) / self.max_yaw:
                chosen["down"] = apf_down
                if chosen["fwd"] - apf_fwd > 2.0:
                    chosen["fwd"] = apf_fwd
            else:
                chosen["yaw"] = apf_yaw
        return chosen


def quantized(action) -> str:
    """Native text of a proposed action (stops excluded: a proposal always moves)."""
    text = encode(action)
    if not decode(text)[2]:
        raise ValueError("A proposal must be a valid quantized action")
    return text


def propose(field, observation, target, rejected_action, collision_source, names):
    """[(name, text)] of the requested proposals at one anchor."""
    rejected = dict(rejected_action)
    apf = field.action(observation, target, rejected, collision_source)
    proposals = {
        "apf": apf,
        "apf_translate": {"fwd": rejected["fwd"], "down": apf["down"], "yaw": float(np.clip(apf["yaw"], -0.224, 0.224))},
        "climb": {"fwd": rejected["fwd"], "down": float(max(-5.0, rejected["down"] - 2.0)), "yaw": rejected["yaw"]},
    }
    return [(name, quantized(proposals[name])) for name in names]
