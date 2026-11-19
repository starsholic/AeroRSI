# Installation

## 1. Environment

Tested on Ubuntu with NVIDIA A800/A100 GPUs, driver 550+, CUDA 12.8 wheels and Python 3.11.

```bash
bash scripts/setup_env.sh
conda activate aerorsi
pytest
```

`scripts/setup_env.sh` creates the conda environment (Python from conda-forge only, so the Anaconda `defaults`
channels are not involved), installs `requirements.txt` with pip, builds `airsim==1.8.1` against the legacy
MessagePack stack, and installs the compiled `msgpack==1.1.2` into `$CONDA_PREFIX/share/aerorsi_rpc`. The `aerorsi`
command loads that copy for the simulator RPCs (see `aerorsi/sim/rpc.py`); without it, image requests take seconds
per step, a known AirSim issue.

The simulator renders off-screen through Vulkan. Check that `/usr/share/vulkan/icd.d/nvidia_icd.json` exists, or
point `AERORSI_VK_ICD` to your NVIDIA ICD file.

## 2. Simulator environments and routes

AeroRSI uses the TravelUAV benchmark exactly as [AeroVLA](https://github.com/XuPeng23/AeroVLA) and
[AeroDPO](https://github.com/XuPeng23/AeroDPO).

1. Download the dataset (`dataset_raw/`) and the simulator environments (`envs/`) following the
   [TravelUAV documentation](https://github.com/prince687028/TravelUAV).
2. Generate the per-route `merged_data.json`, which the raw download does not include, with TravelUAV's tool:

   ```bash
   python tools/generate_merged_json.py --root_dir /path/to/dataset_raw   # inside the TravelUAV repository
   ```

3. Link both into this repository, or set `paths.envs` in `configs/aerorsi.yaml`:

   ```text
   envs/
   ├── carla_town_envs/<Town>/LinuxNoEditor/...
   └── closeloop_envs/...                    # ModernCityDowntown, NewYorkCity, ModularPark, ...
   dataset_raw/<scene>/<route>/{merged_data.json, mark.json, ...}
   ```

4. Build the route manifest. It checks the official splits (`data/splits/`: 7,611 training routes in eleven scenes;
   test seen 1,418, unseen object 629 and unseen map 958) and hashes each route's files:

   ```bash
   aerorsi prepare --dataset-root dataset_raw     # writes outputs/manifest.json
   ```

## 3. Checkpoints

The loop starts from the supervised policy released by AeroDPO; the DPO-init. variant also needs AeroDPO's adapter:

```bash
hf download XuPeng23/AeroDPO --include "AeroVLA-2B-384-Merged/*" --local-dir checkpoints
hf download XuPeng23/AeroDPO --include "AeroDPO-2B-384-LoRA/*" --local-dir checkpoints   # DPO init. only
```

To evaluate a released AeroRSI policy, download its adapter:

```bash
hf download DavidWu1116/AeroRSI --include "AeroRSI-2B-SFTinit/*" --local-dir checkpoints   # or AeroRSI-2B-DPOinit
```

## 4. Resources

A navigation stage runs `--workers-per-gpu` simulator workers on each GPU in `--gpus` (default 4, about 9–10 GB of
GPU memory each, plus two CPU threads). Worker *i* uses RPC port `port_base + i` (default 29200); choose a free range
with `--port-base` if several stages share a machine. Workers start their own simulator processes and stop only those.
