"""Benchmark metrics over evaluated routes: SR, OSR, SPL, NE and CR, by split and by Easy/Hard.

As in the official TravelUAV/AeroDPO script, a route is Easy when its reference trajectory is at most 250 m long and
Hard otherwise; SR counts stops within 20 m of the target, OSR episodes that came within 20 m at any time, SPL weighs
successes by path efficiency against the reference length minus 20 m, NE is the distance from the end of the
reference trajectory, and CR counts depth collisions.
"""
import numpy as np

from .data import TEST_SPLITS

EASY_MAX_M = 250.0


def block(rows):
    if not rows:
        return {"routes": 0}
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    rate = lambda terminal: float(np.mean([r["terminal"] == terminal for r in rows])) * 100
    return {"routes": len(rows), "sr": mean("success") * 100, "osr": mean("oracle_success") * 100,
            "spl": mean("spl") * 100, "ne": mean("ne_m"), "cr": mean("collision") * 100,
            "stuck": rate("stuck"), "wrong_way": rate("wrong_way"), "early_end": rate("early_end"), "timeout": rate("timeout")}


def summarize(results, expected=None):
    """{split: {full, easy, hard}} plus 'all'; `expected` (route ids) marks a run incomplete if routes are missing."""
    ids = [r["route_id"] for r in results]
    if len(ids) != len(set(ids)):
        raise ValueError("A route was evaluated twice")
    summary = {"status": "complete", "routes": len(results)}
    if expected is not None:
        missing = sorted(set(expected) - set(ids))
        if missing:
            summary.update(status="incomplete", missing=missing)
    for name in TEST_SPLITS + ("all",):
        rows = [r for r in results if name == "all" or r["category"] == name]
        summary[name] = {"full": block(rows),
                         "easy": block([r for r in rows if r["reference_length_m"] <= EASY_MAX_M]),
                         "hard": block([r for r in rows if r["reference_length_m"] > EASY_MAX_M])}
    return summary


def table(summary) -> str:
    """Markdown table of the paper's main metrics."""
    lines = ["| Split | Level | Routes | SR | OSR | SPL | NE | CR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in TEST_SPLITS + ("all",):
        for level in ("full", "easy", "hard"):
            b = summary[name][level]
            if b["routes"]:
                lines.append(f"| {name} | {level} | {b['routes']} | {b['sr']:.2f} | {b['osr']:.2f} | {b['spl']:.2f} | "
                             f"{b['ne']:.2f} | {b['cr']:.2f} |")
    return "\n".join(lines)
