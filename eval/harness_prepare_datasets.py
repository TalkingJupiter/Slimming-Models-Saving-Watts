#!/usr/bin/env python
"""Preload lm-eval harness task datasets into the configured HF cache."""

from __future__ import annotations

import argparse

from lm_eval.tasks import TaskManager, get_task_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Preload lm-eval task datasets.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,bbh,arc_challenge",
        help="Comma-separated lm-eval task or group names to preload.",
    )
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task is required")

    task_manager = TaskManager()
    task_dict = get_task_dict(tasks, task_manager=task_manager)
    print(f"[INFO] Prepared {len(task_dict)} lm-eval tasks")
    for task_name in sorted(task_dict):
        print(f"[INFO] Prepared task: {task_name}")


if __name__ == "__main__":
    main()
