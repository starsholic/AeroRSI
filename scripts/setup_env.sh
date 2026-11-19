#!/usr/bin/env bash
# Create the conda environment `aerorsi` and install AeroRSI.
#   bash scripts/setup_env.sh            # uses `conda` from PATH
# Only Python comes from conda, from conda-forge alone (--override-channels, so the Anaconda `defaults` channels and
# their terms of service are not involved); everything else comes from pip (requirements.txt).
# AirSim 1.8.1 must be built against the legacy msgpack stack, so it is installed without build isolation after the
# requirements. The compiled msgpack 1.1.2 that the simulator workers use goes into its own directory
# ($CONDA_PREFIX/share/aerorsi_rpc), which the `aerorsi` command puts first on sys.path (aerorsi/sim/rpc.py).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${AERORSI_ENV:-aerorsi}
conda create -y -n "$ENV" --override-channels -c conda-forge python=3.11 pip
PREFIX=$(conda run -n "$ENV" python -c 'import sys; print(sys.prefix)')
PY="$PREFIX/bin/python"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install 'setuptools<81' wheel
"$PY" -m pip install --no-build-isolation airsim==1.8.1
"$PY" -m pip install --upgrade --target "$PREFIX/share/aerorsi_rpc" msgpack==1.1.2
"$PY" -m pip install -e .
"$PY" -c 'import torch, transformers, peft, airsim, msgpackrpc, cv2; print("OK", torch.__version__, transformers.__version__, peft.__version__)'
echo "Activate with: conda activate $ENV"
