"""TravelUAV routes: the official splits and the route manifest every stage reads.

Training uses the 7,611 official training routes of the eleven seen scenes (the routes AeroDPO mines for
preferences). Evaluation uses the three official test splits: seen (1,418), unseen object (629) and unseen map (958).
The manifest records, for every route, its scene, its trajectory group (routes that share a flown trajectory),
and the hashes of its metadata, so that later stages notice a changed dataset.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from .utils import digest, file_hash, read_json, write_json

DATA = Path(__file__).resolve().parents[1] / "data"
TEST_SPLITS = ("seen", "unseen_object", "unseen_map")
SPAWN_AREAS = DATA / "spawn_areas.json"


def build_manifest(dataset_root, output):
    """Write the manifest of all training and test routes found under the TravelUAV dataset root.

    The dataset root is AeroVLA's ``dataset_raw/``: ``<scene>/<trajectory>/merged_data.json`` and ``mark.json``.
    """
    root = Path(dataset_root).resolve()
    train = read_json(DATA / "splits/train.json")
    test = {name: read_json(DATA / f"splits/test_{name}.json") for name in TEST_SPLITS}
    category = {route_id: name for name, ids in test.items() for route_id in ids}
    if len(train) != 7611 or len(category) != 3005 or set(train) & set(category):
        raise ValueError("Unexpected official split files")
    routes = []
    for route_id in sorted(train + list(category)):
        relative = Path(route_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(route_id)
        metadata, mark = root / relative, root / relative.parent / "mark.json"
        meta = read_json(metadata)
        if "degrees from you." not in meta["conversations"][0]["value"]:
            raise ValueError(f"Unexpected instruction format: {metadata}")
        routes.append({"id": route_id, "scene": relative.parts[0],
                       # routes that share a flown trajectory form one group and never cross a split
                       "group": digest([relative.parts[0], meta["trajectory_raw_detailed"]]),
                       "split": "test" if route_id in category else "train", "category": category.get(route_id, "train"),
                       "metadata": str(metadata), "mark": str(mark),
                       "metadata_sha256": file_hash(metadata), "mark_sha256": file_hash(mark)})
    test_groups = {r["group"] for r in routes if r["split"] == "test"}
    if any(r["group"] in test_groups for r in routes if r["split"] == "train"):
        raise ValueError("A training route shares its trajectory with a test route")
    manifest = {"version": 1, "dataset_root": str(root), "spawn_areas_sha256": file_hash(SPAWN_AREAS),
                "counts": dict(Counter(r["category"] for r in routes)), "routes": routes}
    manifest["identity"] = digest(manifest)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest, overwrite=True)
    return {k: v for k, v in manifest.items() if k != "routes"}


def read_manifest(path):
    manifest = read_json(path)
    identity = manifest.pop("identity")
    if digest(manifest) != identity:
        raise ValueError("Manifest was edited after it was built")
    if file_hash(SPAWN_AREAS) != manifest["spawn_areas_sha256"]:
        raise ValueError("Spawn areas changed after the manifest was built")
    manifest["identity"] = identity
    return manifest


def load_route(record):
    """A manifest record with its instruction, reference trajectory, target position and target object."""
    if file_hash(record["metadata"]) != record["metadata_sha256"] or file_hash(record["mark"]) != record["mark_sha256"]:
        raise ValueError(f"Dataset files changed after the manifest was built: {record['id']}")
    meta, mark = read_json(record["metadata"]), read_json(record["mark"])
    # The target object is placed at the spawn area nearest to the marked target, as in the public evaluator.
    target = np.asarray(mark["target"]["position"])
    areas = [a for a in read_json(SPAWN_AREAS)[record["scene"]] if len(a) >= 18]
    area = min(areas, key=lambda a: np.linalg.norm(np.asarray(a[:3]) + [1, 1, 0.5] - target))
    return {**record, "instruction": meta["conversations"][0]["value"], "trajectory": meta["trajectory_raw_detailed"],
            "target": area[9:12],
            "object": {"position": area[9:12], "orientation": area[13:16] + [area[12]], "scale": area[17], "asset_name": area[16]}}
