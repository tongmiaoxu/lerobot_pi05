import mujoco
from mujoco import viewer
from pathlib import Path

# # Get the path relative to this script's location
# script_dir = Path(__file__).parent
# model_path = str(script_dir.parent / "aloha" / "robolab_setup.xml")
model_path = "./aloha/robolab_setup.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_step(model, data)
viewer.launch(model, data)

import numpy as np

data.ctrl[:] = [0.003067961661145091, -0.9832817316055298, 1.2716701030731201, 0.0, -0.2546408176422119, 0.0076699042692780495, 0.02, -0.0076699042692780495, -0.9848157167434692, 1.2762720584869385, 0.0, -0.2577087879180908, -0.004601942375302315, 0.02]



for _ in range(500):  # 500 * dt steps
    mujoco.mj_step(model, data)

print("qpos (rad):", np.round(data.qpos, 5))
print("ctrl (rad):", np.round(data.ctrl, 5))


qpos_matched = np.concatenate([data.qpos[0:7], data.qpos[8:15]])

err_rad = qpos_matched - data.ctrl
abs_err_rad = np.abs(err_rad)

# Convert to degrees for intuition
rad2deg = 180.0 / np.pi
abs_err_deg = abs_err_rad * rad2deg

# Relative error vs commanded ctrl (avoid divide-by-zero)
ctrl_mag = np.maximum(np.abs(data.ctrl), 1e-6)
rel_err = abs_err_rad / ctrl_mag * 100.0

print("abs error (rad):", np.round(abs_err_rad, 5))
print("abs error (deg):", np.round(abs_err_deg, 2))
print("rel error (% of |ctrl|):", np.round(rel_err, 2))

# import mujoco
# right before left in mujoco, right after left in lerobot
# model = mujoco.MjModel.from_xml_path("aloha/robolab_setup.xml")
# for j in range(model.njnt):
#     print(j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))