"""The policy's native action text and its decoding.

The policy (AeroVLA) answers with three two-digit bins for forward, down and yaw, optionally followed by LAND,
e.g. ``"52 49 49"`` or ``"00 49 00 LAND"``. Each bin maps linearly from 0..98 onto the axis range below. A command
whose three values are all near zero is a stop even without LAND, as in the public evaluator.
"""
from dataclasses import dataclass
import re

import numpy as np

AXES = {"fwd": (0.0, 5.0), "down": (-5.0, 5.0), "yaw": (-1.1, 1.1)}  # metres, metres (positive is down), radians
YAW_MODE_SWITCH = 0.25  # |yaw| at or above this rotates in place; below it the drone translates


@dataclass
class Prediction:
    action: dict
    stop: bool
    text: str
    valid: bool = True


def decode(text: str):
    """(action, stop, valid) with the public parser's clipping and LAND semantics; malformed text is invalid."""
    numbers = re.findall(r"\d+", text)
    action = dict.fromkeys(AXES, 0.0)
    valid = len(numbers) >= 3 or "LAND" in text
    if len(numbers) >= 3:
        action = {key: max(0, min(98, int(n))) / 98 * (hi - lo) + lo for (key, (lo, hi)), n in zip(AXES.items(), numbers)}
    stop = "LAND" in text or (action["fwd"] < 0.01 and abs(action["down"]) < 0.01 and abs(action["yaw"]) < 0.01)
    return action, stop, valid


def encode(action: dict, stop: bool = False) -> str:
    """Native text for an action: three bins, then LAND for a stop (never a bare LAND, which the policy never emits)."""
    values = list(action.values())
    if not np.isfinite(values).all():
        raise ValueError("Non-finite action")
    tokens = " ".join(f"{int(np.rint(np.clip((action[k] - lo) / (hi - lo), 0, 1) * 98)):02d}" for k, (lo, hi) in AXES.items())
    return tokens + " LAND" if stop else tokens


def native_format(text: str) -> bool:
    """Exactly the policy's output format: three two-digit bins, optionally then LAND."""
    parts = text.split()
    numeric = len(parts) in (3, 4) and all(len(p) == 2 and p.isdigit() for p in parts[:3])
    return numeric and (len(parts) == 3 or parts[3] == "LAND")


def mode(text: str) -> str:
    """'invalid', 'LAND', 'turn', 'translate' or 'vertical' (no forward motion)."""
    action, stop, valid = decode(text)
    if not valid:
        return "invalid"
    if stop:
        return "LAND"
    if abs(action["yaw"]) >= YAW_MODE_SWITCH:
        return "turn"
    return "translate" if action["fwd"] > 0.01 else "vertical"


def executed_forward(action: dict) -> float:
    """Forward motion the native controller requests (a turn moves no distance forward)."""
    return float(action["fwd"]) if abs(action["yaw"]) < YAW_MODE_SWITCH else 0.0
