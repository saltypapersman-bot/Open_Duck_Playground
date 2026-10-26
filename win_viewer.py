"""Run the Open Duck MuJoCo viewer on Windows without JAX or mujoco_playground.

The inference script only needs base.get_assets() from the training code, but
importing that module pulls in jax, mjx and mujoco_playground. This launcher
installs lightweight stand-ins for those modules before the import so the
viewer runs on the plain Windows Python (mujoco + onnxruntime + numpy).

Usage:  python win_viewer.py [-o runs/smoke/best_18M.onnx]
Keys in the viewer window: Up/Down walk, Left/Right sidestep, A/E turn, H head mode.
"""
import os
import runpy
import sys
import types

import numpy as np
from etils import epath

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# --- jax stand-in: only used for type annotations and jnp arrays in the viewer path
jax = types.ModuleType("jax")
jax.Array = np.ndarray
jax.numpy = np
sys.modules["jax"] = jax
sys.modules["jax.numpy"] = np

# --- mujoco.mjx stand-in: base.py references mjx.Data / mjx.Model in annotations
import mujoco  # noqa: E402

mjx = types.ModuleType("mujoco.mjx")
mjx.Data = object
mjx.Model = object
mjx.put_model = lambda m: None
mujoco.mjx = mjx
sys.modules["mujoco.mjx"] = mjx

# --- mujoco_playground._src.mjx_env stand-in: MjxEnv base class and update_assets
mp = types.ModuleType("mujoco_playground")
mp_src = types.ModuleType("mujoco_playground._src")
mjx_env = types.ModuleType("mujoco_playground._src.mjx_env")


class MjxEnv:  # base class only; never instantiated by the viewer
    pass


def update_assets(assets, path, glob="*"):
    for f in epath.Path(path).glob(glob):
        if f.is_file():
            assets[f.name] = f.read_bytes()


def get_sensor_data(model, data, sensor_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim]


mjx_env.MjxEnv = MjxEnv
mjx_env.update_assets = update_assets
mjx_env.get_sensor_data = get_sensor_data
sys.modules["mujoco_playground"] = mp
sys.modules["mujoco_playground._src"] = mp_src
sys.modules["mujoco_playground._src.mjx_env"] = mjx_env

if "-o" not in sys.argv and "--onnx_model_path" not in sys.argv:
    sys.argv += ["-o", "runs/smoke/best_18M.onnx"]

runpy.run_path(
    os.path.join(ROOT, "playground", "open_duck_mini_v2", "mujoco_infer.py"),
    run_name="__main__",
)
