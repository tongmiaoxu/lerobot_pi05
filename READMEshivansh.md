Conda env: gello_lerobot

data collection and deploy in real world:
src/lerobot/scripts/lerobot_record.py

visual_match/deploy_act_policy_mujoco.py --color-calibrate --turbo -> Deploy any policy in simulation

data_real -> Eval in real world saved rollouts
data_sim -> Eval in sim saved rollouts

visual_match/load_model_xarm.py -> Load just simulation
Scenes are in xarm7/scene*

Task name and info are in src/lerobot/tasks/task_profiles.py