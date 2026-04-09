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

Minimal workflow:

1. In `src/lerobot/scripts/lerobot_record.py`, set:
   - `_DEFAULT_RECORD_TASK_ID = "pick_mug"` / `"place_mug"` / `"hang_mug"`
   - `_DEFAULT_RECORD_POLICY_CHECKPOINT = None` for teleop collection, or a checkpoint path for deployment
2. Run:

Supported task ids:

- `pick_mug` -> `Pick up the mug`
- `place_mug` -> `Pick and place the mug on the saucer`
- `hang_mug` -> `Hang the mug on the rack`

```bash
lerobot-record
```
Other useful parameters:
```bash
    --dataset.num_episodes=20 \
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
python tools/get_xarm_pointcloud.py --points-per-mesh 10000
```

### Interactive Composite rendering (you can change whether to include the object in the foreground here)
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
python visual_match/compare_recorded_vs_mujoco.py --no-mujoco-view --no_stack
```

### load model in mujoco
```bash
python visual_match/load_model_xarm.py
```

### extract initial states
```bash
python visual_match/initial_states_overlay.py
```

### Policy Training for xarm
`python scripts/train_task.py` auto-fills `dataset.repo_id`, `dataset.root`, `output_dir`, `wandb.enable`, and `wandb.project` from `--task-id`.

```bash
python scripts/train_task.py \
  --task-id=pick_mug \
  --policy-type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --batch_size=8 \
  --steps=80000 \
  --save_freq=5000 \
  --dataset.image_transforms.enable=true
```
```bash
python scripts/train_task.py \
  --task-id=pick_mug \
  --policy-type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --dataset.image_transforms.enable=true \
  --policy.horizon=32 \
  --policy.n_action_steps=25 \
  --batch_size=128 \
  --save_freq=8000 \
  --steps=80000 \
  --config_path=outputs/diffusion_xarm_training/checkpoints/010000/pretrained_model/train_config.json \
  --resume=true
```
```bash
python scripts/train_task.py \
  --task-id=pick_mug \
  --policy-type=pi05 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=50000 \
  --dataset.use_imagenet_stats=false \
  --dataset.image_transforms.enable=true \
  --batch_size=16 \
  --steps=50000 \
  --save_freq=10000
```

```bash
python scripts/train_task.py \
  --task-id=pick_mug \
  --policy-type=groot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.scheduler_decay_steps=90000 \
  --policy.tune_projector=true \
  --dataset.use_imagenet_stats=false \
  --dataset.image_transforms.enable=true \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=20000 \
  --num_workers=4
```

```bash
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 $(which lerobot-train) \ --policy.type=pi05 \ --policy.device=cuda \ --policy.pretrained_path=lerobot/pi05_base \ --policy.push_to_hub=false \ --policy.compile_model=false \ --policy.gradient_checkpointing=true \ --policy.dtype=bfloat16 \ --policy.train_expert_only=true \ --dataset.repo_id=xarm_pick_mug \ --dataset.root=data \ --output_dir=./outputs/pi05_xarm_training \ --batch_size=8 \ --steps=3000 \ --wandb.enable=true \ --wandb.project=pick_mug
```



Defaults in `src/lerobot/scripts/lerobot_record.py`:

- Robot defaults: `xarm_follower` at `192.168.1.228`
- Teleop defaults: `gello_leader` at `/dev/ttyUSB0`
- Dataset defaults: `fps=30`, `push_to_hub=false`, `num_episodes=10`
- Teleop-only recording + `--dataset.task_id=pick_mug`: saves to `data` with repo id `xarm_pick_mug`
- Teleop-only recording + `--dataset.task_id=place_mug`: saves to `data_place_mug` with repo id `place_mug`
- Teleop-only recording + `--dataset.task_id=hang_mug`: saves to `data_hang_mug` with repo id `hang_mug`
- Deployment / evaluation + `--dataset.task_id=pick_mug`: saves to `data_eval` with repo id `eval_xarm_pick_mug`
- Deployment / evaluation + `--dataset.task_id=place_mug`: saves to `data_eval_place_mug` with repo id `eval_place_mug`
- Deployment / evaluation + `--dataset.task_id=hang_mug`: saves to `data_eval_hang_mug` with repo id `eval_hang_mug`

You can still override `--dataset.single_task`, `--dataset.root`, or `--dataset.repo_id` when needed.

### Adjust obj position in mujoco (or --stiker)
```bash
python visual_match/sticker_alpha_calibration.py --cube 
```

### Workflow
Workflow:  gaussian splatting -> point cloud alignment-> camera calibration -> data collection -> composite rendering-> color alignment -> dynamics matching

Run point cloud alignment:(saved to icp_transform.npy)
```bash
python tools/icp_register.py
```
To munaully adjust icp result:
```bash
python tools/VisualizedAlignemnt.py
```
load camera config:
```bash
python tools/load_camera_config.py
```
**adjust object position for color alignment** (press + to increase simulation blending)(arrows with w and s )
```bash
python visual_match/sticker_alpha_calibration.py --mug --sticker --table
```
**Save calibration pairs** from replay (frames 0,1,2,3,4) (press space to pause)
```bash
python visual_match/compare_recorded_vs_mujoco.py --save-calibration-pairs --dataset-path data_color
```
**Run calibration** to learn affine transforms:
```bash
python visual_match/calibrate_color_wrist.py
```
 **Verify**: check `calibration_pairs_wrist/calibrated/combined_*.png` (sim | real | calibrated side-by-side).

 **implement**: color calibration during replay: 
```bash
python visual_match/compare_recorded_vs_mujoco.py --color-calibrate
```
**deployment**:(use --select to select windows for different distributions) (--obs means replace obs with real world images;--obs_eval means replace with real world eval images)(-no_obs means deploy faster)
```bash
python visual_match/deploy_act_policy_mujoco.py --select --gemini
```


```bash
newgrp dialout
conda activate gello_lerobot
```
or
```bash
sudo usermod -aG dialout $USER
reboot
```

```bash
python tools/query_gemini.py -n 1 --stationary
```
```bash
python tools/query_gemini.py -n 3 --wrist
```