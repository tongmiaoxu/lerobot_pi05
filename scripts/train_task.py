#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.tasks import get_task_profile, get_task_profiles


SUPPORTED_POLICIES = ("act", "diffusion", "pi0", "pi05", "groot")


def has_flag(args: list[str], prefix: str) -> bool:
    return any(arg.startswith(prefix) for arg in args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch lerobot-train with task-aware defaults.")
    parser.add_argument("--task-id", required=True, choices=sorted(get_task_profiles()))
    parser.add_argument("--policy-type", required=True, choices=SUPPORTED_POLICIES)
    parser.add_argument("--dry-run", action="store_true", help="Print the final lerobot-train command without running it.")
    args, passthrough = parser.parse_known_args()

    profile = get_task_profile(args.task_id)
    command = ["lerobot-train"]

    default_args = [
        (f"--policy.type={args.policy_type}", "--policy.type="),
        (f"--dataset.repo_id={profile.dataset_repo_id}", "--dataset.repo_id="),
        (f"--dataset.root={profile.dataset_root}", "--dataset.root="),
        (f"--output_dir={profile.output_dir(args.policy_type)}", "--output_dir="),
        ("--wandb.enable=true", "--wandb.enable="),
        (f"--wandb.project={profile.wandb_project}", "--wandb.project="),
    ]
    for value, prefix in default_args:
        if not has_flag(passthrough, prefix):
            command.append(value)

    command.extend(passthrough)
    print(" ".join(shlex.quote(arg) for arg in command))
    if args.dry_run:
        return 0

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
