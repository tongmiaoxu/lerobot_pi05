from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    single_task: str
    dataset_repo_id: str
    dataset_root: str
    eval_dataset_repo_id: str
    eval_dataset_root: str
    sim_eval_dataset_root: str
    selection_object_name: str
    wandb_project: str
    output_dirs_by_policy: dict[str, str] = field(default_factory=dict)

    def output_dir(self, policy_type: str) -> str:
        if policy_type in self.output_dirs_by_policy:
            return f"outputs/{self.output_dirs_by_policy[policy_type]}"
        return f"outputs/{policy_type}_{self.task_id}"

    def _policy_suffix(self, policy_type: str | None = None, checkpoint_name: str | None = None) -> str:
        parts = []
        if policy_type:
            parts.append(policy_type)
        if checkpoint_name:
            parts.append(checkpoint_name)
        return "" if not parts else "_" + "_".join(parts)

    def eval_root_for_policy(
        self, policy_type: str | None = None, checkpoint_name: str | None = None
    ) -> str:
        return f"{self.eval_dataset_root}{self._policy_suffix(policy_type, checkpoint_name)}"

    def sim_eval_root_for_policy(
        self, policy_type: str | None = None, checkpoint_name: str | None = None
    ) -> str:
        return f"{self.sim_eval_dataset_root}{self._policy_suffix(policy_type, checkpoint_name)}"


DEFAULT_TASK_ID = "pick_mug"

_TASK_PROFILES: dict[str, TaskProfile] = {
    "pick_mug": TaskProfile(
        task_id="pick_mug",
        single_task="Pick up the mug",
        dataset_repo_id="xarm_pick_mug",
        dataset_root="data",
        eval_dataset_repo_id="eval_xarm_pick_mug",
        eval_dataset_root="data_eval",
        sim_eval_dataset_root="data_sim_eval",
        selection_object_name="mug",
        wandb_project="pick_mug",
        output_dirs_by_policy={
            "act": "act_xarm_training",
            "diffusion": "diffusion_xarm_training",
            "pi05": "pi05_xarm_training",
            "groot": "groot_xarm_training",
        },
    ),
    "place_mug": TaskProfile(
        task_id="place_mug",
        single_task="Pick and place the mug on the saucer",
        dataset_repo_id="place_mug",
        dataset_root="data_place_mug",
        eval_dataset_repo_id="eval_place_mug",
        eval_dataset_root="data_eval_place_mug",
        sim_eval_dataset_root="data_sim_eval_place_mug",
        selection_object_name="mug, saucer",
        wandb_project="place_mug",
    ),
    "hang_mug": TaskProfile(
        task_id="hang_mug",
        single_task="Hang the mug on the rack",
        dataset_repo_id="hang_mug",
        dataset_root="data_hang_mug",
        eval_dataset_repo_id="eval_hang_mug",
        eval_dataset_root="data_eval_hang_mug",
        sim_eval_dataset_root="data_sim_eval_hang_mug",
        selection_object_name="mug, rack",
        wandb_project="hang_mug",
    ),
}


def get_task_profiles() -> dict[str, TaskProfile]:
    return dict(_TASK_PROFILES)


def get_task_profile(task_id: str) -> TaskProfile:
    try:
        return _TASK_PROFILES[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(_TASK_PROFILES))
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of: {available}.") from exc
