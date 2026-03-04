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
### Camera calibration
1. go to /home/tina/Documents/residual_physics-main
```bash
conda activate calib
pip install opencv-contrib-python
python experiments/real_world/calibrate.py --calibrate
```
2. in this codebase, run: (stationary cam, wrist cam)
```bash
python visual_match/load_calibration_to_config.py 311322300308 213622251153
```

### Command (xArm + GELLO data collection)

```bash
# Cameras auto-loaded from visual_match/configs/ (stationary_cam.json -> cam_high, wrist_cam.json -> cam_wrist)
# To disable: --robot.cameras='{}'
lerobot-record \
    --robot.type=xarm_follower \
    --robot.ip=192.168.1.228 \
    --teleop.type=gello_leader \
    --teleop.port=/dev/ttyUSB0 \
    --dataset.repo_id=tongmiao/xarm_pick_cube \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.root=data \
    --resume=true
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
in simulation: 
```bash
lerobot-teleoperate --robot.type=xarm_sim_follower --robot.launch_sim=true --teleop.type=gello_leader --teleop.port=/dev/ttyUSB0 --fps=100
```

### GELLO Calibration (get offsets)
save offsets to `.cache/calibration/gello_leader/gello_offsets.json`.
They are loaded automatically by `lerobot-record` and `lerobot-teleoperate`.
— Use gello_software's offsets directly** ( --use-gello-software-offsets):
Default xarm pose: `[0, 0, 0, 90, 0, 90, 0]` degrees — joints 4 and 6 at 90°, rest at 0°.
```bash
python tools/gello_get_offset.py --port /dev/ttyUSB0 --use-gello-software-offsets
```

### Point Cloud of xArm (ground truth from MuJoCo model → PCD)
```bash
python tools/get_xarm_pointcloud.py --points-per-mesh 100000
```

### Interactive Composite rendering
```bash
python visual_match/composite_rendering.py 
```

### replay in mujoco (--cma : use cma result in cma_result.pkl )
```bash
python visual_match/run_prerecorded_traj_mujoco.py --cma
```

### compare replay
```bash
python visual_match/compare_recorded_vs_mujoco.py --cma
```

```bash
python visual_match/compare_recorded_vs_mujoco.py --no-mujoco-view
```

### load model in mujoco
```bash
python visual_match/load_model_xarm.py
```

### Policy Training for xarm
```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=tongmiao/xarm_pick_cube \
  --dataset.root=data \
  --output_dir=./outputs/act_xarm_training \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --batch_size=8 \
  --steps=80000 \
  --wandb.enable=true \
  --wandb.project=xarm_pick_mug_lerobot0.4.3 \
  --dataset.image_transforms.enable=true
```
```bash
lerobot-train \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=tongmiao/xarm_pick_cube \
  --dataset.root=data \
  --output_dir=./outputs/diffusion_xarm_training \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --wandb.enable=true \
  --wandb.project=xarm_pick_mug_lerobot0.4.3  \
  --dataset.image_transforms.enable=true \
  --policy.horizon=64 \
  --policy.n_action_steps=50 \
  --batch_size=64 \
  --steps=7000
```
```bash
lerobot-train \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --dataset.repo_id=tongmiao/xarm_pick_cube \
  --dataset.root=data \
  --output_dir=./outputs/pi05_xarm_training \
  --batch_size=8 \
  --steps=10000 \
  --wandb.enable=true \
  --wandb.project=xarm_pick_mug_lerobot0.4.3
```
```bash
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 $(which lerobot-train) \ --policy.type=pi05 \ --policy.device=cuda \ --policy.pretrained_path=lerobot/pi05_base \ --policy.push_to_hub=false \ --policy.compile_model=false \ --policy.gradient_checkpointing=true \ --policy.dtype=bfloat16 \ --policy.train_expert_only=true \ --dataset.repo_id=tongmiao/xarm_pick_cube \ --dataset.root=data \ --output_dir=./outputs/pi05_xarm_training \ --batch_size=8 \ --steps=3000 \ --wandb.enable=true \ --wandb.project=xarm_pick_mug_lerobot0.4.3
```

### Policy Deployment for xarm
```bash
lerobot-record \
  --robot.type=xarm_follower \
  --robot.ip=192.168.1.228 \
  --teleop.type=gello_leader \
  --teleop.port=/dev/ttyUSB0 \
  --policy.path=outputs/pi05_xarm_training/checkpoints/001000/pretrained_model \
  --dataset.repo_id=tongmiao/eval_xarm_pick_cube \
  --dataset.single_task="Pick up the cube" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --dataset.root=data_eval \
  --dataset.push_to_hub=false \
  --policy.pretrained_path=lerobot/pi05_base
```
### Policy rollout in Simulation for xarm (--obs means replace obs with real world images)
```bash
python visual_match/deploy_act_policy_mujoco.py --obs
```
### Adjust obj position in mujoco (or --stiker)
```bash
python visual_match/sticker_alpha_calibration.py --cube 
```

### Wrist color calibration (sim → real)
1. **Save calibration pairs** from replay (frames 0,5,10,15,20):
```bash
python visual_match/compare_recorded_vs_mujoco.py --save-calibration-pairs
```
2. **Run calibration** to learn affine transforms:
```bash
python visual_match/calibrate_color_wrist.py
```
3. **Verify**: check `calibration_pairs_wrist/calibrated/combined_*.png` (sim | real | calibrated side-by-side).

4. **implement**: color calibration during replay: 
```bash
python visual_match/compare_recorded_vs_mujoco.py --color-calibrate
```
5.**deployment**:
```bash
python visual_match/deploy_act_policy_mujoco.py --color-calib-path
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
