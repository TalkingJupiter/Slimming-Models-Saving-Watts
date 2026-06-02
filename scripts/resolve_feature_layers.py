#!/usr/bin/env python3
"""Resolve depth-normalized teacher/student layers for feature KD."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DEPTH_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")


def safe_hf_model_name(model: str) -> str:
    return model.replace("/", "_")


def config_path_for(model: str, project_root: str) -> Optional[Path]:
    if not model:
        return None

    direct = Path(model) / "config.json"
    if direct.is_file():
        return direct

    local = Path(project_root) / ".hf_models" / safe_hf_model_name(model) / "config.json"
    if local.is_file():
        return local

    return None


def load_depth(model: str, project_root: str, role: str) -> int:
    cfg_path = config_path_for(model, project_root)
    if cfg_path is not None:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in DEPTH_KEYS:
            value = cfg.get(key)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(f"{role} config has no supported depth key in {cfg_path}")

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
        value = getattr(cfg, "num_hidden_layers", None)
        if isinstance(value, int) and value > 0:
            return value
    except Exception as exc:
        raise ValueError(
            f"Could not resolve {role} depth for {model!r}. Warm the model locally under "
            f"$PROJECT_ROOT/.hf_models or pass an explicit layer override. Original error: {exc}"
        ) from exc

    raise ValueError(f"Could not resolve {role} depth for {model!r}")


def layer_from_ratio(depth: int, ratio: float) -> int:
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
    return max(0, min(depth - 1, int(((depth - 1) * ratio) + 0.5)))


def parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", default="")
    parser.add_argument("--ratio", type=float, default=0.60)
    parser.add_argument("--teacher-layer", default=None)
    parser.add_argument("--student-layer", default=None)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", os.getcwd()))
    args = parser.parse_args()

    teacher_override = parse_optional_int(args.teacher_layer)
    student_override = parse_optional_int(args.student_layer)

    teacher_depth = load_depth(args.teacher, args.project_root, "teacher")
    teacher_layer = teacher_override if teacher_override is not None else layer_from_ratio(teacher_depth, args.ratio)
    if teacher_layer < 0 or teacher_layer >= teacher_depth:
        raise ValueError(f"teacher layer {teacher_layer} outside valid range 0..{teacher_depth - 1}")

    student_depth = 0
    student_layer = student_override
    if args.student:
        student_depth = load_depth(args.student, args.project_root, "student")
        if student_layer is None:
            student_layer = layer_from_ratio(student_depth, args.ratio)
        if student_layer < 0 or student_layer >= student_depth:
            raise ValueError(f"student layer {student_layer} outside valid range 0..{student_depth - 1}")

    print(f"FEATURE_LAYER_RATIO={args.ratio}")
    print(f"FEATURE_TEACHER_DEPTH={teacher_depth}")
    print(f"FEATURE_TEACHER_LAYER={teacher_layer}")
    if args.student:
        print(f"FEATURE_STUDENT_DEPTH={student_depth}")
        print(f"FEATURE_STUDENT_LAYER={student_layer}")


if __name__ == "__main__":
    main()
