from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    single_task: str
    dataset_repo_id: str
    dataset_root: str
    dataset_root_480640: str
    eval_dataset_repo_id: str
    eval_dataset_root: str
    sim_eval_dataset_root: str
    selection_object_name: str
    wandb_project: str
    scene_xml_candidates: tuple[str, ...] = ("scene.xml",)
    xarm7_xml: str = "xarm7.xml"
    deploy_adjustable_object_names: tuple[str, ...] = ("mug",)
    calibration_adjustable_object_names: tuple[str, ...] = ("mug",)
    #: MuJoCo `(body_name, free_joint_name)` pairs for objects adjusted via `data.qpos` in calibration.
    calibration_free_joint_pairs: tuple[tuple[str, str], ...] = ()
    #: Non-free bodies whose orientation is tracked for manual yaw (and XML euler writes).
    calibration_body_yaw_rotatable_names: tuple[str, ...] = ()
    #: Body whose free-joint pose matches trailing qpos in `<key name="home">`; None = infer if exactly one pair.
    calibration_xarm7_home_free_joint_body: str | None = None
    output_dirs_by_policy: dict[str, str] = field(default_factory=dict)

    def calibration_free_joint_pair_dict(self) -> dict[str, str]:
        return dict(self.calibration_free_joint_pairs)

    def calibration_xarm7_home_free_joint_body_resolved(self) -> str | None:
        if self.calibration_xarm7_home_free_joint_body is not None:
            return self.calibration_xarm7_home_free_joint_body
        pairs = self.calibration_free_joint_pairs
        return pairs[0][0] if len(pairs) == 1 else None

    def output_dir(self, policy_type: str) -> str:
        if policy_type in self.output_dirs_by_policy:
            return f"outputs/{self.output_dirs_by_policy[policy_type]}"
        return f"outputs/{policy_type}_{self.task_id}"

    def calibration_pairs_dir(self, camera_name: str = "stationary") -> Path:
        return Path(self.dataset_root_480640) / f"calibration_pairs_{camera_name}"

    def color_calibration_path(self, camera_name: str = "stationary") -> Path:
        return self.calibration_pairs_dir(camera_name) / "calibrated" / "color_mapping.yaml"

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
        dataset_root="data_pick_mug",
        dataset_root_480640 = "data_pick_mug_copy",
        eval_dataset_repo_id="eval_xarm_pick_mug",
        eval_dataset_root="data_eval_pick_mug",
        sim_eval_dataset_root="data_sim_eval_pick_mug",
        selection_object_name="mug",
        wandb_project="pick_mug",
        scene_xml_candidates=("scene.xml",),
        deploy_adjustable_object_names=("mug",),
        calibration_adjustable_object_names=("mug", "sticker"),
        calibration_free_joint_pairs=(("mug", "mug_joint"),),
        calibration_body_yaw_rotatable_names=(),
        output_dirs_by_policy={
            "act": "act_xarm_training",
            "diffusion": "diffusion_xarm_training",
            "pi0": "pi0_xarm_training",
            "pi05": "pi05_xarm_training",
            "groot": "groot_xarm_training",
        },
    ),
    "place_mug": TaskProfile(
        task_id="place_mug",
        single_task="Pick and place the mug on the saucer",
        dataset_repo_id="place_mug",
        dataset_root="data_place_mug",
        dataset_root_480640 = "data_place_mug_copy",
        eval_dataset_repo_id="eval_place_mug",
        eval_dataset_root="data_eval_place_mug",
        sim_eval_dataset_root="data_sim_eval_place_mug",
        selection_object_name="mug, saucer",
        wandb_project="place_mug",
        scene_xml_candidates=("scene_saucer.xml",),
        xarm7_xml="xarm7_saucer.xml",
        deploy_adjustable_object_names=("mug", "saucer"),
        calibration_adjustable_object_names=("mug", "saucer","table"),
        calibration_free_joint_pairs=(("mug", "mug_joint"),),
        calibration_body_yaw_rotatable_names=("saucer",),
    ),
    "hang_mug": TaskProfile(
        task_id="hang_mug",
        single_task="Hang the mug on the rack",
        dataset_repo_id="hang_mug",
        dataset_root="data_hang_mug",
        dataset_root_480640 = "data_hang_mug_copy",
        eval_dataset_repo_id="eval_hang_mug",
        eval_dataset_root="data_eval_hang_mug",
        sim_eval_dataset_root="data_sim_eval_hang_mug",
        selection_object_name="mug",
        wandb_project="hang_mug",
        scene_xml_candidates=("scene_hang.xml",),
        xarm7_xml="xarm7_hang.xml",
        deploy_adjustable_object_names=("mug",),
        calibration_adjustable_object_names=("mug", "rack","table"),
        calibration_free_joint_pairs=(("mug", "mug_joint"),),
        calibration_body_yaw_rotatable_names=("rack",),
    ),
    "pick_shoe": TaskProfile(
        task_id="pick_shoe",
        single_task="Pick up the shoe",
        dataset_repo_id="xarm_pick_shoe",
        dataset_root="data_pick_shoe",
        dataset_root_480640="data_pick_shoe_copy",
        eval_dataset_repo_id="eval_xarm_pick_shoe",
        eval_dataset_root="data_eval_pick_shoe",
        sim_eval_dataset_root="data_sim_eval_pick_shoe",
        selection_object_name="right_shoe",
        wandb_project="pick_shoe",
        scene_xml_candidates=("scene_shoe.xml",),
        xarm7_xml="xarm7_shoe.xml",
        deploy_adjustable_object_names=("right_shoe",),
        calibration_adjustable_object_names=("left_shoe", "right_shoe", "table"),
        calibration_free_joint_pairs=(("right_shoe", "right_shoe_joint"),),
        calibration_body_yaw_rotatable_names=("left_shoe",),
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


def resolve_task_scene_xml(task_id: str, xarm_dir: str | Path) -> Path:
    task_profile = get_task_profile(task_id)
    xarm_dir = Path(xarm_dir)
    for scene_name in task_profile.scene_xml_candidates:
        scene_path = xarm_dir / scene_name
        if scene_path.exists():
            return scene_path

    searched = ", ".join(str(xarm_dir / scene_name) for scene_name in task_profile.scene_xml_candidates)
    raise FileNotFoundError(f"Could not find a scene XML for task_id {task_id!r}. Checked: {searched}")


def resolve_task_xarm7_xml(task_id: str, xarm_dir: str | Path) -> Path:
    task_profile = get_task_profile(task_id)
    path = Path(xarm_dir) / task_profile.xarm7_xml
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find robot XML for task_id {task_id!r}: {path} "
            f"(expected filename from task profile: {task_profile.xarm7_xml!r})"
        )
    return path
