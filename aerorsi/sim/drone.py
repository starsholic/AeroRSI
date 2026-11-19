"""The TravelUAV/AeroVLA drone interface on one AirSim client.

These functions issue the same RPC sequences as the upstream ``AirVLNSimulatorClientTool`` (AeroVLA/AeroDPO), without
its one-thread-per-client wrapper. ``move`` is the native action controller and defines the action semantics of the
benchmark; it must not be changed.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("FrontCamera", "LeftCamera", "RightCamera", "RearCamera", "DownCamera")
RGB_CAMERAS = ("FrontCamera", "DownCamera")


def kinematics(state):
    """airsim.KinematicsState from a saved state dict (position, velocities, accelerations, x-y-z-w orientation)."""
    import airsim
    kin = airsim.KinematicsState()
    for name in ("position", "linear_velocity", "linear_acceleration", "angular_velocity", "angular_acceleration"):
        setattr(kin, name, airsim.Vector3r(*state.get(name, [0.0, 0.0, 0.0])))
    kin.orientation = airsim.Quaternionr(*state["orientation"])
    return kin


def read_state(client):
    data = client.getMultirotorState()
    collision = client.simGetCollisionInfo()
    k = data.kinematics_estimated
    return {"collision": {"has_collided": collision.has_collided, "object_name": data.collision.object_name},
            "gps_location": [data.gps_location.latitude, data.gps_location.longitude, data.gps_location.altitude],
            "timestamp": data.timestamp, "position": list(k.position), "linear_velocity": list(k.linear_velocity),
            "linear_acceleration": list(k.linear_acceleration), "orientation": list(k.orientation),
            "angular_velocity": list(k.angular_velocity), "angular_acceleration": list(k.angular_acceleration)}


def read_imu(client, imu_name=""):
    data = client.getImuData(imu_name=imu_name)
    q0, q1, q2, q3 = data.orientation.w_val, data.orientation.x_val, data.orientation.y_val, data.orientation.z_val
    rotation = [[1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0), 2 * (q1 * q3 + q2 * q0)],
                [2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q1 * q0)],
                [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2)]]
    return {"time_stamp": data.time_stamp, "rotation": rotation, "orientation": list(data.orientation),
            "linear_acceleration": list(data.linear_acceleration), "angular_velocity": list(data.angular_velocity)}


def sensors(client):
    return {"state": read_state(client), "imu": read_imu(client)}


def set_kinematics(client, state):
    """Write a full kinematic state and advance one frame, leaving the simulation paused."""
    client.simSetKinematics(state=kinematics(state), ignore_collision=True)
    client.simContinueForFrames(1)
    client.simPause(True)


def spawn_object(client, index, asset_name, pose, scale):
    """Spawn the target object as my_object_<index>, replacing the previous one."""
    if index > 0:
        client.simDestroyObject(f"my_object_{index - 1}")
    success = client.simSpawnObject(f"my_object_{index}", asset_name, pose, scale, physics_enabled=False, is_blueprint=False)
    client.simContinueForFrames(1)
    client.simPause(True)
    return success


def images(client, retries=10):
    """(rgb, depth): RGB of the front and down cameras (zeros for the others) and uint8 depth of all five
    cameras, depth clipped to 100 m and scaled to 0..255."""
    import airsim
    requests = []
    for camera in CAMERAS:
        if camera in RGB_CAMERAS:
            requests.append(airsim.ImageRequest(camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
        requests.append(airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
    for attempt in range(retries + 1):
        try:
            responses = iter(client.simGetImages(requests=requests))
            rgb, depth = [], []
            for camera in CAMERAS:
                if camera in RGB_CAMERAS:
                    r = next(responses)
                    rgb.append(np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3))
                else:
                    rgb.append(np.zeros((64, 64, 3), dtype=np.uint8))
                r = next(responses)
                metres = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
                depth.append((np.clip(metres, 0, 100) / 100 * 255).astype(np.uint8))
            return rgb, depth
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1)


def move(client, action, start_state):
    """Execute one action with the native controller; returns the sensors read after the closing hover.

    The drone first turns to the commanded heading. Below |yaw| 0.25 rad it then translates by (fwd, down) along
    that heading at 1 m/s (moves under 1 m take one second); at larger |yaw| it only changes altitude. Every action
    ends with a 0.5 s zero-velocity hold.
    """
    import airsim
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(False)
    q = start_state.orientation
    yaw = Rotation.from_quat([q.x_val, q.y_val, q.z_val, q.w_val]).as_euler("zyx")[0] + action["yaw"]
    dx, dy, dz = action["fwd"] * np.cos(yaw), action["fwd"] * np.sin(yaw), action["down"]
    total = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    epsilon = 0.01
    if np.hypot(dx, dy) > epsilon or abs(dz) > epsilon or abs(action["yaw"]) > epsilon:
        client.rotateToYawAsync(np.degrees(yaw), timeout_sec=3, margin=5).join()
        if abs(action["yaw"]) < 0.25:
            if total > epsilon:
                if total < 1.0:
                    duration, vx, vy, vz = 1.0, dx, dy, dz
                else:
                    duration, vx, vy, vz = total, dx / total, dy / total, dz / total
                client.moveByVelocityAsync(vx, vy, vz, duration=duration, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                           yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=np.degrees(yaw))).join()
        else:
            client.moveToZAsync(start_state.position.z_val + dz, velocity=2.0, timeout_sec=3).join()
    client.moveByVelocityAsync(0, 0, 0, duration=0.5).join()
    return {"state": read_state(client), "imu": read_imu(client, "Imu")}
