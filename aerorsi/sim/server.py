"""Launch one headless TravelUAV scene on one GPU and RPC port, owned and cleaned up by the calling worker.

Unlike the upstream launcher, nothing here kills other processes: each worker starts its own Unreal process from
a private runtime directory (hard links and symlinks into the shared environment) and terminates only that process.
"""
from contextlib import ExitStack, contextmanager
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

SETTINGS = Path(__file__).with_name("airsim_settings.json")
CLOSELOOP_BINARIES = {"ModernCityMap": "ModernCityDowntown", "NewYorkCity": "NewYorkCity",
                      "NYCEnvironmentMegapa": "NYCEnvironmentMegapa", "TropicalIsland": "TropicalIsland",
                      "ModularPark": "ModularPark", "ModularEuropean": "ModularEuropean"}
STARTUP_SECONDS = 48  # wait at least this long after the RPC port opens, before the first request
VULKAN_ICD = os.environ.get("AERORSI_VK_ICD", "/usr/share/vulkan/icd.d/nvidia_icd.json")


def scene_binary(envs: Path, scene: str):
    """(source directory, binary name) of a scene in the TravelUAV environment download."""
    if scene.startswith("Carla_"):
        return envs / "carla_town_envs" / scene[len("Carla_"):] / "LinuxNoEditor", "CarlaUE4"
    return envs / "closeloop_envs", CLOSELOOP_BINARIES[scene]


class Simulator:
    def __init__(self, envs, scene, output, gpu, port):
        self.envs, self.scene, self.output = Path(envs).resolve(), scene, Path(output).resolve()
        self.gpu, self.port = gpu, port
        self.process = self.stream = self.object_client = None

    def __enter__(self):
        import airsim
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"RPC port {self.port} is already in use")
        source, name = scene_binary(self.envs, self.scene)
        root = self.output / "runtime"
        root.mkdir(parents=True)
        (root / "Engine").symlink_to(source / "Engine", target_is_directory=True)
        (root / name / "Binaries/Linux").mkdir(parents=True)
        (root / name / "Content").symlink_to(source / name / "Content", target_is_directory=True)
        binary = root / name / "Binaries/Linux" / name
        # Unreal locates Engine/ and Content/ from the path of its own binary, so the binary must live in the runtime
        # directory (a symlink would resolve back to the shared install): hard-link it, or copy it across filesystems.
        try:
            os.link(source / name / "Binaries/Linux" / name, binary)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source / name / "Binaries/Linux" / name, binary)
        settings = json.loads(SETTINGS.read_text())
        settings["ApiServerPort"] = self.port
        settings_path = self.output / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))
        command = [str(binary), name, "-RenderOffscreen", "-NoSound", "-NoVSync", "-unattended",
                   f"-GraphicsAdapter={self.gpu}", f"-settings={settings_path}", f"-abslog={self.output / 'unreal.log'}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu), VK_ICD_FILENAMES=VULKAN_ICD, SDL_VIDEODRIVER="dummy",
                   __GL_SHADER_DISK_CACHE_PATH=str(self.output / "shader_cache"))
        for key in ("DISPLAY", "DRI_PRIME", "VK_INSTANCE_LAYERS"):
            env.pop(key, None)
        self.stream = (self.output / "process.log").open("x")
        try:
            started = time.monotonic()
            self.process = subprocess.Popen(command, cwd=root, env=env, stdout=self.stream, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            while time.monotonic() - started < 180:
                if self.process.poll() is not None:
                    raise RuntimeError("Simulator exited during startup")
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                        break
                time.sleep(0.5)
            else:
                raise TimeoutError("Simulator did not open its RPC port within 180 s")
            time.sleep(max(0.0, STARTUP_SECONDS - (time.monotonic() - started)))
            self._check_placement()
            self.client = airsim.MultirotorClient(port=self.port, timeout_value=20)
            if not self.client.ping():
                raise RuntimeError("AirSim RPC ping failed")
            self._extend_object_timeouts(airsim.MultirotorClient)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _check_placement(self):
        """The Unreal process must render on its assigned GPU (it may also hold a context on GPU 0)."""
        tree = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
        devices = [i for i, gpu in enumerate(tree.findall("gpu")) for p in gpu.findall("processes/process_info")
                   if int(p.findtext("pid", "-1")) == self.process.pid]
        if self.gpu not in devices or not set(devices) <= {0, self.gpu}:
            raise RuntimeError(f"Unexpected simulator GPU placement: {devices}")

    def _extend_object_timeouts(self, factory):
        """Object spawning can take minutes on a cold scene: give those two RPCs their own client with a long timeout,
        after checking that the game thread answers."""
        self.object_client = factory(port=self.port, timeout_value=180)
        if not isinstance(self.object_client.simListSceneObjects("^__aerorsi_readiness_probe__$"), list):
            raise RuntimeError("Invalid AirSim readiness response")
        self.client.simSpawnObject = self.object_client.simSpawnObject
        self.client.simDestroyObject = self.object_client.simDestroyObject

    def __exit__(self, *_):
        if self.object_client is not None:
            try:
                self.object_client.client.close()
            except Exception:
                pass
            self.object_client = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.stream:
            self.stream.close()


@contextmanager
def start_simulator(envs, scene, output, gpu, port, attempts=3):
    """Start a scene, retrying only an Unreal segmentation fault during startup (before any reset or action)."""
    output = Path(output)
    with ExitStack() as stack:
        for attempt in range(1, attempts + 1):
            simulator = Simulator(envs, scene, output, gpu, port)
            try:
                ready = stack.enter_context(simulator)
                break
            except RuntimeError:
                code = simulator.process.poll() if simulator.process is not None else None
                logs = [p for p in (output / "process.log", output / "unreal.log") if p.is_file()]
                crashed = code in (-11, 139) and any("SIGSEGV" in p.read_text(errors="replace")
                                                     or "Signal 11 caught" in p.read_text(errors="replace") for p in logs)
                if not crashed or attempt == attempts:
                    raise
                output.rename(output.with_name(f"{output.name}_startup_failed_{attempt:02d}"))
        yield ready
