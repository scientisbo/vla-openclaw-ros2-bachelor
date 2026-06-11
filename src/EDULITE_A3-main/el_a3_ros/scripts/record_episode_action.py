#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


RECORD_TOPICS = [
    "/joint_states",
    "/dynamic_joint_states",
    "/tf",
    "/tf_static",
    "/arm_controller/controller_state",
    "/gripper_controller/controller_state",
    "/display_planned_path",
]


def utc_now() -> str:
    """返回带时区的 UTC 时间，用于跨设备统一比较日志时间。"""
    return datetime.now(timezone.utc).isoformat()


def local_timestamp() -> str:
    """返回用于 episode 文件夹名称的本地时间。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_and_save(
    command: list[str],
    output_path: Path,
    timeout: float = 10.0,
) -> int:
    """执行诊断命令，并把 stdout/stderr 保存到文件。"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        text = result.stdout

        if result.stderr:
            text += "\n[stderr]\n" + result.stderr

        output_path.write_text(text, encoding="utf-8")
        return result.returncode

    except Exception as exc:
        output_path.write_text(
            f"Command failed: {command}\n"
            f"Exception: {exc!r}\n",
            encoding="utf-8",
        )
        return -1


def create_episode_directories(data_root: Path) -> tuple[str, Path]:
    episode_id = f"episode_{local_timestamp()}_action"
    episode_dir = data_root / episode_id

    for name in [
        "rosbag",
        "meta",
        "analysis",
        "logs",
        "screenshots",
    ]:
        (episode_dir / name).mkdir(parents=True, exist_ok=True)

    return episode_id, episode_dir


def start_rosbag(
    episode_dir: Path,
) -> tuple[subprocess.Popen, object]:
    bag_dir = episode_dir / "rosbag" / "el_a3_action"
    log_path = episode_dir / "logs" / "rosbag_record.log"

    command = [
        "ros2",
        "bag",
        "record",
        "-o",
        str(bag_dir),
        *RECORD_TOPICS,
    ]

    log_file = log_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    return process, log_file


def stop_process_group(
    process: subprocess.Popen,
    timeout: float = 10.0,
) -> int:
    """
    优先发送 SIGINT，让 rosbag 正常写完 metadata 和数据库索引。
    若长时间未退出，再发送 SIGTERM。
    """
    if process.poll() is not None:
        return process.returncode

    try:
        os.killpg(process.pid, signal.SIGINT)
        return process.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)

        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return process.wait()


def run_executor(
    ros_root: Path,
    task_plan_path: Path,
    skill_log_path: Path,
    executor_log_path: Path,
) -> int:
    executor_script = ros_root / "scripts" / "run_skill_executor_action.py"

    command = [
        sys.executable,
        str(executor_script),
        "--task-plan",
        str(task_plan_path),
        "--log",
        str(skill_log_path),
    ]

    with executor_log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=ros_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return result.returncode


def count_skill_results(skill_log_path: Path) -> tuple[int, int]:
    success_count = 0
    failed_count = 0

    if not skill_log_path.exists():
        return success_count, failed_count

    for line in skill_log_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("event") != "skill_finished":
            continue

        if event.get("status") == "success":
            success_count += 1
        elif event.get("status") == "failed":
            failed_count += 1

    return success_count, failed_count


def generate_summary(
    episode_dir: Path,
    metadata: dict,
    success_count: int,
    failed_count: int,
) -> None:
    summary_path = episode_dir / "analysis" / "episode_summary.md"

    lines = [
        "# Episode Summary",
        "",
        f"- Episode ID: `{metadata['episode_id']}`",
        f"- Instruction: {metadata['instruction']}",
        f"- Mode: `{metadata['mode']}`",
        f"- Status: `{metadata['status']}`",
        f"- Started at: `{metadata['started_at']}`",
        f"- Finished at: `{metadata['finished_at']}`",
        f"- Successful skills: `{success_count}`",
        f"- Failed skills: `{failed_count}`",
        f"- Executor return code: `{metadata['executor_returncode']}`",
        f"- Rosbag detected as valid: `{metadata['rosbag_valid']}`",
        "",
        "## Recorded topics",
        "",
    ]

    for topic in metadata["record_topics"]:
        lines.append(f"- `{topic}`")

    lines.extend(
        [
            "",
            "## Result",
            "",
        ]
    )

    if metadata["status"] == "success":
        lines.append(
            "The structured task plan was executed through ROS2 "
            "FollowJointTrajectory actions and robot state topics "
            "were recorded by rosbag2."
        )
    else:
        lines.append(
            "The episode did not finish successfully. Inspect "
            "`logs/executor.log`, `logs/rosbag_record.log`, "
            "`logs/controllers.txt`, and the skill execution log."
        )

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record one EL-A3 Action execution episode."
        )
    )

    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural-language task instruction.",
    )

    parser.add_argument(
        "--task-plan",
        required=True,
        help="Existing task_plan.json produced by the planner.",
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
    )

    parser.add_argument(
        "--post-record-seconds",
        type=float,
        default=2.0,
        help="Continue recording briefly after executor finishes.",
    )

    args = parser.parse_args()

    ros_root = Path(__file__).resolve().parents[1]
    data_root = ros_root / "data" / "raw"

    source_task_plan = Path(args.task_plan).expanduser().resolve()

    if not source_task_plan.is_file():
        print(
            f"[ERROR] task plan not found: {source_task_plan}",
            file=sys.stderr,
        )
        return 2

    episode_id, episode_dir = create_episode_directories(data_root)

    task_plan_path = episode_dir / "meta" / "task_plan.json"
    skill_log_path = (
        episode_dir / "meta" / "skill_execution_log.jsonl"
    )
    replan_log_path = episode_dir / "meta" / "replan_log.jsonl"

    shutil.copy2(source_task_plan, task_plan_path)
    replan_log_path.touch()

    metadata = {
        "episode_id": episode_id,
        "instruction": args.instruction,
        "mode": args.mode,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "record_topics": RECORD_TOPICS,
        "task_plan_path": str(task_plan_path),
        "skill_execution_log_path": str(skill_log_path),
        "rosbag_path": str(
            episode_dir / "rosbag" / "el_a3_action"
        ),
    }

    metadata_path = episode_dir / "meta" / "metadata.json"
    write_json(metadata_path, metadata)

    # 保存运行前的 ROS 图状态。
    run_and_save(
        ["ros2", "node", "list"],
        episode_dir / "logs" / "node_list.txt",
    )
    run_and_save(
        ["ros2", "topic", "list"],
        episode_dir / "logs" / "topic_list.txt",
    )
    run_and_save(
        ["ros2", "action", "list"],
        episode_dir / "logs" / "action_list.txt",
    )
    run_and_save(
        ["ros2", "control", "list_controllers"],
        episode_dir / "logs" / "controllers.txt",
    )

    print(f"[episode] directory: {episode_dir}")
    print("[episode] starting rosbag2...")

    rosbag_process: Optional[subprocess.Popen] = None
    rosbag_log_file = None
    executor_returncode = -1
    rosbag_returncode = -1

    try:
        rosbag_process, rosbag_log_file = start_rosbag(
            episode_dir
        )

        # 给 rosbag2 时间发现 topics 并完成订阅。
        time.sleep(2.0)

        if rosbag_process.poll() is not None:
            raise RuntimeError(
                "rosbag2 exited before executor started"
            )

        print("[episode] running Action executor...")

        executor_returncode = run_executor(
            ros_root=ros_root,
            task_plan_path=task_plan_path,
            skill_log_path=skill_log_path,
            executor_log_path=(
                episode_dir / "logs" / "executor.log"
            ),
        )

        time.sleep(max(0.0, args.post_record_seconds))

    except KeyboardInterrupt:
        print("\n[episode] interrupted by user")

    except Exception as exc:
        print(f"[episode] exception: {exc!r}", file=sys.stderr)

        (
            episode_dir / "logs" / "wrapper_exception.txt"
        ).write_text(
            repr(exc) + "\n",
            encoding="utf-8",
        )

    finally:
        if rosbag_process is not None:
            print("[episode] stopping rosbag2...")
            rosbag_returncode = stop_process_group(
                rosbag_process
            )

        if rosbag_log_file is not None:
            rosbag_log_file.close()

    bag_dir = episode_dir / "rosbag" / "el_a3_action"

    bag_info_returncode = run_and_save(
        ["ros2", "bag", "info", str(bag_dir)],
        episode_dir / "logs" / "rosbag_info.txt",
        timeout=20.0,
    )

    rosbag_valid = (
        bag_info_returncode == 0
        and (bag_dir / "metadata.yaml").is_file()
    )

    success_count, failed_count = count_skill_results(
        skill_log_path
    )

    status = (
        "success"
        if executor_returncode == 0 and rosbag_valid
        else "failed"
    )

    metadata.update(
        {
            "status": status,
            "finished_at": utc_now(),
            "executor_returncode": executor_returncode,
            "rosbag_returncode": rosbag_returncode,
            "rosbag_valid": rosbag_valid,
            "successful_skills": success_count,
            "failed_skills": failed_count,
        }
    )

    write_json(metadata_path, metadata)

    generate_summary(
        episode_dir=episode_dir,
        metadata=metadata,
        success_count=success_count,
        failed_count=failed_count,
    )

    print(f"[episode] status: {status}")
    print(
        "[episode] summary: "
        f"{episode_dir / 'analysis' / 'episode_summary.md'}"
    )

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
