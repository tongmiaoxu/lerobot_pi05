python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=pick_cuber_v30 \
    --dataset.root=data \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=tongmiao/pi05_pick_cube \
    --policy.push_to_hub=true \
    --output_dir=./outputs/pi05_pick_cube \
    --job_name=pi05_pick_cube \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --batch_size=4 \
    --steps=3000 \
    --wandb.enable=true \
    --wandb.project=pi05_aloha

    # Check GPU memory
nvidia-smi

# Policy evaluation (cameras are now default in aloha_follower config)
lerobot-record \
    --robot.type=aloha_follower \
    --teleop.type=bi_aloha_leader \
    --policy.path=/home/tongmiao/Documents/lerobot_pi05/outputs/pi05_pick_cube/checkpoints/last/pretrained_model \
    --dataset.repo_id=tongmiao/eval_pi05_aloha \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=false 2>&1 | tee eval_log.txt

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

### Policy Training
```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=tongmiao/aloha_pick_cube \
  --dataset.root=data/tongmiao/aloha_pick_cube \
  --output_dir=./outputs/act_training \
  --batch_size=32 \
  --steps=40000
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
python visual_match/run_prerecorded_traj_mujoco.py     --dataset-path data/  --episode 0 --new
```
### Compare Recorded vs MuJoCo (--new means using new normalization method in lerobot0.4.3)

```bash
python visual_match/compare_recorded_vs_mujoco.py --dataset-path data/ --episode 0 --new 
 
```
### Policy rollout
```bash
python visual_match/deploy_act_policy_mujoco.py \
    --policy-path outputs/train_alohacodebase/act_pick_cuber/checkpoints/080000/pretrained_model \
    --prompt "Pick up the cube" \
    --fps 30
```


