
## Data Collection
src/lerobot/scripts/lerobot_record.py    ← Main entry point (lerobot-record command)
├── src/lerobot/robots/aloha_follower/   ← Robot implementation
│   ├── aloha_follower.py                  (AlohaFollower class)
│   ├── aloha_arm.py                       (Single arm implementation)
│   └── config_aloha_follower.py           (Robot config)
├── src/lerobot/teleoperators/aloha_leader/  ← Teleoperator implementation
│   ├── bi_aloha_leader.py                 (BiAlohaLeader class)
│   ├── aloha_leader_arm.py                (Single arm implementation)
│   └── config_aloha_leader.py             (Teleop config)
├── src/lerobot/datasets/lerobot_dataset.py  ← Dataset creation & saving
├── src/lerobot/utils/control_utils.py       ← Keyboard listener
└── .cache/calibration/                      ← Calibration files
### Command (ALOHA-style state machine)
```bash
# Data collection with default cameras (cam_high, cam_low, cam_left_wrist, cam_right_wrist): the defalut dataset path is -/home/tongmiao/.cache/huggingface/lerobot/{repo_id}: Episode data: all episodes are in data/chunk-000/file-000.parquet
lerobot-record \
    --robot.type=aloha_follower \
    --teleop.type=bi_aloha_leader \
    --dataset.repo_id=tongmiao/aloha_pick_cube \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.root=data 

```
or some other parameters:
```bash
    --resume=true \
    --dataset.push_to_hub=false \
    --display_data=true
```

---

## Data Collection (xArm + GELLO)
src/lerobot/scripts/lerobot_record.py    ← Main entry point (lerobot-record command)
├── src/lerobot/robots/xarm_follower/     ← Robot implementation
│   ├── xarm_follower.py                   (XarmFollower class, uses xArm SDK)
│   └── config_xarm_follower.py            (Robot config: IP, DOF, gripper, cameras)
├── src/lerobot/teleoperators/gello_leader/ ← Teleoperator implementation
│   ├── gello_leader.py                    (GelloLeader class, Dynamixel XL330)
│   └── config_gello_leader.py             (Teleop config: port, DOF, offsets)
├── src/lerobot/datasets/lerobot_dataset.py  ← Dataset creation & saving
├── src/lerobot/utils/control_utils.py       ← Keyboard listener
└── .cache/calibration/                      ← Calibration files

### Prerequisites
```bash
pip install xarm-python-sdk
```

### Command (xArm + GELLO data collection)
```bash
# Basic data collection with xArm7 + GELLO (no cameras):
lerobot-record \
    --robot.type=xarm_follower \
    --robot.ip=192.168.1.228 \
    --robot.dof=7 \
    --teleop.type=gello_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.dof=7 \
    --dataset.repo_id=tongmiao/xarm_pick_cube \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.root=data
```
```bash
# With 2 RealSense cameras (stationary + wrist):
# To find serial numbers, run: lerobot-find-cameras realsense
lerobot-record \
    --robot.type=xarm_follower \
    --robot.ip=192.168.1.228 \
    --robot.cameras='{"cam_high": {"type": "intelrealsense", "serial_number_or_name": "246322303954", "fps": 30, "width": 640, "height": 480}, "cam_wrist": {"type": "intelrealsense", "serial_number_or_name": "213622251153", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=gello_leader \
    --teleop.port=/dev/ttyUSB0 \
    --dataset.repo_id=tongmiao/xarm_pick_cube \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.root=data
```
Other useful parameters:
```bash
    --resume=true \
    --dataset.push_to_hub=false \
    --display_data=true
```

### Teleoperation only (xArm + GELLO, no data recording)
```bash
lerobot-teleoperate --robot.type=xarm_follower --teleop.type=gello_leader --robot.ip=192.168.1.228 --teleop.port=/dev/ttyUSB0 --fps=100
```

### GELLO Calibration (get offsets)
Same approach as gello_software's calibration.

**Option A — Use gello_software's offsets directly** (recommended if your GELLO already works with gello_software):
```bash
python scripts/gello_get_offset.py --port /dev/ttyUSB0 --use-gello-software-offsets
```

**Option B — Compute offsets from scratch**:

1. Place GELLO so it matches the xArm's known start pose.
   Default: `[0, 0, 0, 90, 0, 90, 0]` degrees — joints 4 and 6 at 90°, rest at 0°.
   Leave the GELLO gripper fully open.
2. Run:
```bash
python scripts/gello_get_offset.py --port /dev/ttyUSB0
```
3. Verify the `corrected` values match `expected`. If they don't, check GELLO alignment.

Both options save offsets to `.cache/calibration/gello_leader/gello_offsets.json`.
They are loaded automatically by `lerobot-record` and `lerobot-teleoperate`.

---

### Policy Training
```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=tongmiao/aloha_pick_cube \
  --dataset.root=data/tongmiao/aloha_pick_cube \
  --output_dir=./outputs/act2_training_wrist \
  --policy.image_keys_filter='["cam_right_wrist", "cam_left_wrist"]' \
  --batch_size=8 \
  --steps=80000 \
  --wandb.enable=true \
  --wandb.project=aloha_pick_cube_lerobot0.4.3_wrist \
  --dataset.image_transforms.enable=true

```
```bash
lerobot-train \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=tongmiao/aloha_pick_cube \
  --dataset.root=data/tongmiao/aloha_pick_cube \
  --output_dir=./outputs/diffusion_training_wrist \
  --policy.image_keys_filter='["cam_right_wrist", "cam_left_wrist"]' \
  --wandb.enable=true \
  --wandb.project=aloha_pick_cube_lerobot0.4.3_wrist \
  --dataset.image_transforms.enable=true \
  --policy.horizon=64 \
  --policy.n_action_steps=50 \
  --batch_size=64
```
### Policy Deployment
```bash
lerobot-record \
  --robot.type=aloha_follower \
  --teleop.type=bi_aloha_leader \
  --policy.path=outputs/act_training_wrist/checkpoints/last/pretrained_model \
  --dataset.repo_id=tongmiao/eval_act_aloha \
  --dataset.single_task="Pick up the cube" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.root=data_eval \
  --dataset.push_to_hub=false \
  --resume=true

```




### Teleoperation (no data recording)
To config which arm to use: edit `config_aloha_follower.py` (line 48-49) and `config_aloha_leader.py` (line 68-69)
```bash
lerobot-teleoperate \
    --robot.type=aloha_follower \
    --teleop.type=bi_aloha_leader \
    --fps=30
```
### Load model
```bash
python visual_match/load_model.py
```
### Replay in mujoco (--new means using new normalization method in lerobot0.4.3)
```bash
python visual_match/run_prerecorded_traj_mujoco.py     --dataset-path data/tongmiao/aloha_pick_cube/ --episode 0 --new
```

### Policy rollout in Simulation
Color calibration is applied automatically if calibration_pairs_wrist/calibrated/color_mapping.yaml exists.

```bash
python visual_match/deploy_act_policy_mujoco.py \
    --policy-path outputs/act_training_wrist/checkpoints/last/pretrained_model \
    --prompt "Pick up the cube" \
    --fps 30 \
    --new

python visual_match/deploy_act_policy_mujoco.py \
    --policy-path outputs/diffusion_training_wrist/checkpoints/last/pretrained_model \
    --prompt "Pick up the cube" \
    --fps 30 \
    --new
```
```bash
python visual_match/deploy_act_policy_mujoco.py \
    --policy-path outputs/train_alohacodebase/act_pick_cuber/checkpoints/080000/pretrained_model \
    --prompt "Pick up the cube" \
    --fps 30
```

### Compare Recorded vs MuJoCo (--new means using new normalization method in lerobot0.4.3)
Color calibration is applied automatically if calibration_pairs_wrist/calibrated/color_mapping.yaml exists.

```bash
python visual_match/compare_recorded_vs_mujoco.py --dataset-path data/tongmiao/aloha_pick_cube/ --episode 0 --new --color-calibrate --alpha 0.6 --save-images
```

```bash
python visual_match/compare_recorded_vs_mujoco.py --dataset-path data/tongmiao/aloha_pick_cube/ --episode 0 --new --alpha 0.6
```



```bash
newgrp dialout
conda activate gello_lerobot
```
or
```bash
sudo usermod -aG dialout $USER
reboot
```bash
