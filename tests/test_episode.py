import numpy as np

from aerorsi.actions import Prediction
from aerorsi.episode import EpisodeStatus, depth_collision

MOVE, STOP = Prediction({"fwd": 5.0, "down": 0.0, "yaw": 0.0}, False, "98 49 49"), Prediction({"fwd": 0.0, "down": 0.0, "yaw": 0.0}, True, "00 49 49")


def observation(x, depth=255):
    return {"sensors": {"state": {"position": [x, 0.0, -10.0]}}, "depth": [np.full((8, 8), depth, np.uint8)] * 5}


def test_depth_collision_needs_ten_percent_of_one_camera():
    obs = observation(0)
    assert depth_collision(obs) == ""
    obs["depth"] = [np.full((8, 8), 255, np.uint8)] * 4 + [np.zeros((8, 8), np.uint8)]
    assert depth_collision(obs) == "downcamera_depth"


def test_success_requires_a_stop_within_20_m():
    status = EpisodeStatus(target=[100.0, 0.0, -10.0])
    status.start(observation(0))
    for x in (30, 60, 85):
        status.update(observation(x), MOVE)
    assert not status.done and status.oracle_success
    status.update(observation(85), STOP)
    assert status.done and status.success and status.terminal == "success"


def test_early_stop_does_not_end_unless_oracle():
    status = EpisodeStatus(target=[100.0, 0.0, -10.0])
    status.start(observation(0))
    status.update(observation(10), STOP)
    assert status.early_end and not status.done


def test_wrong_way_after_ten_non_decreasing_distances():
    status = EpisodeStatus(target=[100.0, 0.0, -10.0])
    status.start(observation(50))
    for i in range(9):
        status.update(observation(50 - 2 * (i + 1)), MOVE)
    assert status.terminal == "wrong_way"


def test_collision_ends_the_episode():
    status = EpisodeStatus(target=[100.0, 0.0, -10.0])
    status.start(observation(0))
    status.update(observation(5, depth=0), MOVE)
    assert status.done and status.terminal == "collision"
