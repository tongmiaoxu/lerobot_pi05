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