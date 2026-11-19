"""A restorable navigation world: one owned simulator, reset to a route start, stepped by native actions.

Two stepping cadences are used. Evaluation steps ``paused=False`` exactly like the public evaluator. Collection
(rollouts of training routes and repair verification) pauses the simulation after every action, so that the state
before each action is a complete kinematic snapshot that ``restore`` can reapply later to fly alternative actions.
"""
from contextlib import ExitStack
import copy
import time

import numpy as np

from . import drone
from .rpc import compiled_rpc_transport
from .server import start_simulator

RESET = {"stabilizing_seconds": 0.5, "pose_writes": 3, "attempts": 3}
# A restored or reset state must match its target within these tolerances, or the result is discarded.
TOLERANCES = {"position": 0.1, "linear_velocity": 0.2, "angular_velocity": 0.1, "rotation_radians": 0.035}


class InfrastructureError(RuntimeError):
    """A simulator or RPC failure; never counted as a navigation outcome."""


def state_errors(expected, observed):
    errors = {k: float(np.linalg.norm(np.asarray(expected[k]) - observed[k]))
              for k in ("position", "linear_velocity", "angular_velocity")}
    q1, q2 = np.asarray(expected["orientation"]), np.asarray(observed["orientation"])
    denominator = np.linalg.norm(q1) * np.linalg.norm(q2)
    errors["rotation_radians"] = (float(2 * np.arccos(np.clip(abs(np.dot(q1, q2) / denominator), 0, 1)))
                                  if denominator > 0 else float("inf"))
    return {k: v if np.isfinite(v) else None for k, v in errors.items()}


def state_matches(errors):
    return all(errors[k] is not None and errors[k] <= tolerance for k, tolerance in TOLERANCES.items())


class World:
    def __init__(self, slot, output, envs):
        self.slot, self.output, self.envs = slot, output, envs
        self.stack, self.scene_stack = ExitStack(), None
        self.scene, self.scene_index, self.object_index = None, 0, 0
        self.last_physical_collision = False
        self.last_reset_diagnostics = self.last_restore_errors = None

    def __enter__(self):
        self.stack.enter_context(compiled_rpc_transport())
        return self

    def __exit__(self, *args):
        if self.scene_stack:
            self.scene_stack.close()
        self.stack.close()

    # -- episode start ---------------------------------------------------------------------------------------------
    def reset(self, route):
        """Load the route's scene if needed, spawn its target object and place the drone at the route start."""
        import airsim
        self.last_reset_diagnostics = None
        if route["scene"] != self.scene:
            if self.scene_stack:
                self.scene_stack.close()
            self.scene_stack = ExitStack()
            folder = self.output / f"scene_{self.scene_index:03d}_{route['scene']}"
            folder.mkdir()
            self.scene_index += 1
            simulator = self.scene_stack.enter_context(
                start_simulator(self.envs, route["scene"], folder / "simulator", self.slot["gpu"], self.slot["port"]))
            self.client, self.scene, self.object_index = simulator.client, route["scene"], 0
        obj = route["object"]
        pose = airsim.Pose(airsim.Vector3r(*obj["position"]), airsim.Quaternionr(*obj["orientation"]))
        # As upstream, the spawn RPC's return value is not trusted; the object's pose is checked instead.
        drone.spawn_object(self.client, self.object_index, obj["asset_name"], pose, airsim.Vector3r(*[obj["scale"]] * 3))
        actual = self.client.simGetObjectPose(f"my_object_{self.object_index}")
        self.object_index += 1
        if not np.isfinite(list(actual.position)).all():
            raise InfrastructureError("Target object absent after spawn")
        # Spawning advances one frame, so the drone is placed afterwards.
        sensors = self._place(route["trajectory"][0])
        self.last_physical_collision = False
        return self.observe(sensors)

    def _place(self, first):
        """Arm, hover and write the full start state; retried up to three times if the state does not match."""
        start = {k: [0.0, 0.0, 0.0] for k in ("linear_velocity", "linear_acceleration", "angular_velocity", "angular_acceleration")}
        start.update(position=list(first["position"]), orientation=list(first["orientation"]))
        attempts = []
        for _ in range(RESET["attempts"]):
            self.client.simPause(True)
            self.client.enableApiControl(True)
            if not self.client.armDisarm(True):
                raise InfrastructureError("Arming failed")
            self.client.simSetKinematics(drone.kinematics(start), ignore_collision=True)
            # A joined zero-velocity command replaces the previous controller task and stabilizes the vehicle.
            self.client.simPause(False)
            self.client.moveByVelocityAsync(0, 0, 0, duration=RESET["stabilizing_seconds"]).join()
            self.client.simPause(True)
            for _ in range(RESET["pose_writes"]):
                drone.set_kinematics(self.client, start)
            sensors = drone.sensors(self.client)
            errors = state_errors(start, sensors["state"])
            self.last_reset_diagnostics = {"expected": start, "errors": errors, "valid": state_matches(errors),
                                           "prior_attempts": attempts}
            if state_matches(errors):
                return sensors
            attempts.append(errors)
        raise InfrastructureError(f"Start state mismatch after {RESET['attempts']} attempts: {attempts}")

    # -- stepping --------------------------------------------------------------------------------------------------
    def observe(self, sensors=None):
        rgb, depth = drone.images(self.client)
        if any(not np.isfinite(d).all() for d in depth):
            raise InfrastructureError("Non-finite depth")
        self.observation = {"rgb": rgb, "depth": depth, "sensors": sensors if sensors is not None else drone.sensors(self.client),
                            "physical_collision": self.last_physical_collision}
        return self.observation

    def snapshot(self):
        if not self.client.simIsPause():
            raise RuntimeError("Snapshots need the paused collection cadence")
        return copy.deepcopy(self.observation["sensors"]["state"])

    def step(self, action, *, paused):
        previous = self.client.simGetCollisionInfo().time_stamp
        after = drone.move(self.client, action, drone.kinematics(self.observation["sensors"]["state"]))
        if paused:
            self.client.simPause(True)
        collision = self.client.simGetCollisionInfo()
        self.last_physical_collision = bool(collision.has_collided and collision.time_stamp > previous)
        time.sleep(0.01)  # the public evaluator's observation delay
        return self.observe(None if paused else after)

    def restore(self, state):
        """Reapply a saved paused state; returns whether the observed state matches it within TOLERANCES."""
        # Rendering needs an advancing frame: flush the previous task, advance one frame, then rewrite the state.
        self.client.simPause(True)
        self.client.cancelLastTask()
        self.client.simSetKinematics(drone.kinematics(state), ignore_collision=True)
        self.client.simContinueForFrames(1)
        self.client.simPause(True)
        for _ in range(3):
            self.client.simSetKinematics(drone.kinematics(state), ignore_collision=True)
        self.last_physical_collision = False
        self.last_restore_errors = state_errors(state, self.observe()["sensors"]["state"])
        return state_matches(self.last_restore_errors)
