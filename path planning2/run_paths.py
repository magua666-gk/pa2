"""Helpers for organizing training and test artifacts by run."""

import datetime
import os
from typing import Dict, Iterable, Optional


def ensure_dir_exists(dir_path: str) -> str:
    """Create a directory if needed and return it as a string."""
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(tag: Optional[str], default_tag: str = "experiment") -> str:
    """Sanitize user-provided tags for safe filesystem paths."""
    if not tag:
        return default_tag

    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(tag).strip())
    return safe or default_tag


def build_run_id(seed: Optional[int] = None, run_id: Optional[int] = None, timestamp: Optional[str] = None) -> str:
    """Build a stable run id that distinguishes repeated train/test executions."""
    parts = [timestamp or get_timestamp()]
    if run_id is not None:
        parts.append(f"run{run_id}")
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def _join_config_parts(config_parts: Optional[Iterable[str]]) -> str:
    if not config_parts:
        return ""
    cleaned = [
        sanitize_tag(str(part).strip("_"), default_tag="")
        for part in config_parts
        if str(part).strip("_")
    ]
    cleaned = [part for part in cleaned if part]
    return "_" + "_".join(cleaned) if cleaned else ""


def build_result_roots(output_root: str, experiment_tag: str) -> Dict[str, str]:
    """Return root folders for one experiment family."""
    safe_tag = sanitize_tag(experiment_tag)
    result_dir = os.path.join(output_root, "results", safe_tag)
    training_base = os.path.join(result_dir, "training")
    tests_base = os.path.join(result_dir, "tests")

    for path in (result_dir, training_base, tests_base):
        ensure_dir_exists(path)

    return {
        "experiment_tag": safe_tag,
        "result_dir": result_dir,
        "training_base": training_base,
        "test_results_base": tests_base,
    }


def build_training_run_paths(
    output_root: str,
    experiment_tag: str,
    seed: Optional[int] = None,
    run_id: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """Create model/result folders for a single training run."""
    safe_tag = sanitize_tag(experiment_tag)
    base_run_stamp = build_run_id(seed=seed, run_id=run_id, timestamp=timestamp)
    run_stamp = base_run_stamp

    result_dir = os.path.join(output_root, "results", safe_tag)
    model_root = os.path.join(output_root, "models", safe_tag)
    training_root = os.path.join(result_dir, "training")
    counter = 2
    while (
        os.path.exists(os.path.join(model_root, run_stamp))
        or os.path.exists(os.path.join(training_root, run_stamp))
    ):
        run_stamp = f"{base_run_stamp}_{counter:02d}"
        counter += 1

    model_dir = os.path.join(model_root, run_stamp)
    training_dir = os.path.join(training_root, run_stamp)
    plots_dir = os.path.join(training_dir, "plots")
    diagnostics_dir = os.path.join(training_dir, "diagnostics")
    logs_dir = os.path.join(training_dir, "logs")

    for path in (model_dir, result_dir, training_dir, plots_dir, diagnostics_dir, logs_dir):
        ensure_dir_exists(path)

    return {
        "experiment_tag": safe_tag,
        "run_stamp": run_stamp,
        "model_dir": model_dir,
        "result_dir": result_dir,
        "training_dir": training_dir,
        "plots_dir": plots_dir,
        "diagnostics_dir": diagnostics_dir,
        "logs_dir": logs_dir,
        "final_model_path": os.path.join(model_dir, "final_model"),
        "leader_model_path": os.path.join(model_dir, "Path_SAC_actor_L1.pth"),
        "follower_model_path": os.path.join(model_dir, "Path_SAC_actor_F1.pth"),
        "training_result_path": os.path.join(training_dir, "training_results.pkl"),
        "training_plot_path": os.path.join(plots_dir, "training_curve.png"),
        "diagnostic_path": os.path.join(diagnostics_dir, "training_diagnostics.json"),
    }


def build_test_run_dir(
    tests_base: str,
    run_kind: str,
    config_parts: Optional[Iterable[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """Create a directory for one test run."""
    safe_kind = sanitize_tag(run_kind, default_tag="test")
    run_stamp = timestamp or get_timestamp()
    base_test_dir_name = f"{safe_kind}_{run_stamp}{_join_config_parts(config_parts)}"
    test_dir_name = base_test_dir_name
    counter = 2
    while os.path.exists(os.path.join(tests_base, test_dir_name)):
        test_dir_name = f"{base_test_dir_name}_{counter:02d}"
        counter += 1
    test_dir = os.path.join(tests_base, test_dir_name)
    ensure_dir_exists(test_dir)
    return {
        "run_stamp": run_stamp,
        "test_dir_name": test_dir_name,
        "test_dir": test_dir,
    }
