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
Other useful parameters: (--obs meaning taking dataset obs[imgae + states])
```bash
    --dataset.num_episodes=20 \
    --resume=true \
    --dataset.push_to_hub=false \
    --display_data=true \
    --obs=true
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
  --task-id=place_mug \
  --policy-type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --batch_size=32 \
  --steps=90000 \
  --save_freq=30000 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --dataset.image_transforms.enable=true
```
```bash
python scripts/train_task.py \
  --task-id=hang_mug \
  --policy-type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_keys_filter='["cam_high", "cam_wrist"]' \
  --dataset.image_transforms.enable=true \
  --policy.horizon=56 \
  --policy.n_action_steps=50 \
  --batch_size=128 \
  --save_freq=30000 \
  --steps=90000 \
  --config_path=outputs/diffusion_xarm_training/checkpoints/010000/pretrained_model/train_config.json \
  --resume=true
```
(diffusion n_obs_steps = 2;n_action_steps <= horizon - n_obs_steps + 1; horizon divisible by 8)


```bash
python scripts/train_task.py \
  --task-id=place_mug \
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
  --steps=100000 \
  --save_freq=40000
```

```bash
python scripts/train_task.py \
  --task-id=place_mug \
  --policy-type=pi0 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi0_base \
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
  --save_freq=5000
```

```bash
python scripts/train_task.py \
  --task-id=place_mug \
  --policy-type=groot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.scheduler_decay_steps=100000 \
  --policy.tune_projector=true \
  --dataset.use_imagenet_stats=false \
  --dataset.image_transforms.enable=true \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=40000 \
  --num_workers=4 \
  --dataset.video_backend=pyav
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

**Run calibration** to learn affine transforms:
```bash
python visual_match/calibrate_color.py
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

Go to task_profiles.py to edit task configs.

Collect dataset:
Set _DEFAULT_RECORD_POLICY_CHECKPOINT = None in lerobot-record
```bash
python tools/gello_get_offset.py --port /dev/ttyUSB0
lerobot-record
```
Make a copy before DS

DS
```bash
python scripts/prepare_lerobot_dataset_224.py /path/to/your_dataset_name
```

Train
Copy dataset to cluster

Initial Dist:
set DATA_DIR = copy before DS
set TEXT_PROMPTS
```bash
python visual_match/initial_states_overlay.py
python visual_match/temp_pixel_masks_overlay_10ep.py
```
Saved under DATA_DIR.

Align object for replay or color alignment:
(--task-id place_mug   --episode 24   --save-auto-align-cache to overwrite the obj alignment optimization cache with manual adjustment; or give multiple episodes: --episode 0-4,95-99 gives autosave mode)
(python visual_match/compare_recorded_vs_mujoco.py   --episode 24   --save-replay-frames   --auto-align-force(to recompute even cache exist))
m        select mug
Arrows   move selected object in X/Y
w / s    move selected object up/down in Z
j / l    rotate yaw left/right
x        reset editable poses
+ / -    adjust alpha blend
Space    replay from calibration frame
q        save, or save and next episode in range mode
Esc      discard/stop
```bash
python tools/decomp_collision_mujoco.py
python visual_match/obj_calibration_mujoco.py
```

Replay:
```bash
python visual_match/compare_recorded_vs_mujoco.py
```

Color Alignment:
( --save-replay-frames: save every single frame under root/gs_render/episode_*/stationary/frame_XXXX.png and {root}/gs_render/episode_*/wrist/frame_XXXX.png; 
--auto-align-initial-objects)
```bash
python visual_match/compare_recorded_vs_mujoco.py --save-calibration-pairs
python visual_match/calibrate_color.py
```
Deployment:
(--auto-align-initial-objects)
Set policy checkpoint, run
```bash
lerobot-record
python visual_match/deploy_act_policy_mujoco.py
```
Then deployment selection defaults come from the task_profiles.py:
pick_mug -> mug
place_mug -> mug + saucer
hang_mug -> mug + rack

```bash
python sim2real/train.py \
  --sim-dir data_place_mug_copy/calibration_pairs_stationary/gs_renders \
  --real-dir data_place_mug_copy/calibration_pairs_stationary/real_captures \
  --dataset-dir data_transfer_pairs \
  --output-dir outputs/turbo_sim2real --overwrite \
  --max-train-steps 2002 \
  --learning-rate 1e-4 \
  --train-batch-size 1 \
  --val-ratio 0.0 \
  --gradient-checkpointing \
  --enable-xformers \
  --mixed-precision no \
  --resolution 224
```
```bash

python sim2real/train.py \
  --sim-dir data_place_mug_copy/gs_render \
  --real-dir data_place_mug_copy/real_captures \
  --camera wrist \
  --dataset-dir data_transfer_pairs_wrist \
  --output-dir outputs/turbo_sim2real_wrist \
  --overwrite \
  --max-train-steps 10002 \
  --learning-rate 1e-4 \
  --val-ratio 0.1 \
  --viz-freq 200 \
  --eval-freq 200 \
  --checkpointing-steps 1000 \
  --lambda-gan 0.5 \
  --lambda-clipsim 0 \
  --lambda-dinov3-pixel 1.0 \
  --lambda-l2 0 \
  --lambda-lpips 0 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --mixed-precision bf16 \
  --resultion 224 \
  --pair-selection odd \
  --max-pairs 268
```
```bash
python -m sim2real.img2img_turbo.inference_paired \
  --model_path outputs/turbo_sim2real/checkpoints/model_2001.pkl \
  --input_image data_place_mug_copy/calibration_pairs_stationary/gs_renders/frame_0000.png \
  --prompt "a real-world robot camera image" \
  --output_dir data_place_mug_copy/calibration_pairs_stationary/ \
  --resolution 224 \
  --use_fp16
```
```bash
python -m sim2real.img2img_turbo.inference_paired \
  --model_path outputs/turbo_sim2real_wrist_dino/checkpoints/model_30001.pkl \
  --input_image data_place_mug_copy/gs_render/episode_000011/wrist \
  --prompt "a real-world robot camera image" \
  --resolution 224 \
  --use_fp16
```
```bash( --fast-rollout-video-replay: only render the camera after all deployments)
python visual_match/deploy_act_policy_mujoco.py \
  --turbo \
  --turbo-checkpoint-stationary outputs/turbo_sim2real_stationary_dino_hang/checkpoints/model_30001.pkl \
  --turbo-checkpoint-wrist outputs/turbo_sim2real_wrist_dino_hang/checkpoints/model_30001.pkl \
  --fast-rollout-video-replay
```
```bash
python tools/query_gpt_image.py \
  --input-image data_place_mug_copy/gs_render/stationary/frame_0000.png \
  --style-image data_place_mug_copy/real_captures/stationary/frame_0000.png \
  --prompt "Transfer style while preserving geometry."
```