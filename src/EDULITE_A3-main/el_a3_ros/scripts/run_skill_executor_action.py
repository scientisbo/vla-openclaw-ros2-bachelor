#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINTS = [
    "L1_joint",
    "L2_joint",
    "L3_joint",
    "L4_joint",
    "L5_joint",
    "L6_joint",
]

GRIPPER_JOINTS = [
    "L7_joint",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class ActionSkillExecutor(Node):
    def __init__(self, log_path: Path):
        super().__init__("action_skill_executor")
        self.log_path = log_path

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )

    def wait_for_servers(self) -> bool:
        self.get_logger().info("Waiting for arm_controller action server...")
        arm_ok = self.arm_client.wait_for_server(timeout_sec=5.0)

        self.get_logger().info("Waiting for gripper_controller action server...")
        gripper_ok = self.gripper_client.wait_for_server(timeout_sec=5.0)

        return arm_ok and gripper_ok

    def send_trajectory(
        self,
        *,
        action_client: ActionClient,
        controller_name: str,
        joint_names: list[str],
        positions: list[float],
        duration_sec: float,
        timeout_sec: float,
        skill_name: str,
    ) -> bool:
        if len(joint_names) != len(positions):
            raise ValueError(
                f"{skill_name}: joint_names length {len(joint_names)} "
                f"!= positions length {len(positions)}"
            )

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        append_jsonl(
            self.log_path,
            {
                "timestamp": now_iso(),
                "event": "action_goal_send",
                "skill": skill_name,
                "controller": controller_name,
                "joint_names": joint_names,
                "positions": positions,
                "duration_sec": duration_sec,
            },
        )

        send_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout_sec)

        if not send_future.done():
            append_jsonl(
                self.log_path,
                {
                    "timestamp": now_iso(),
                    "event": "action_goal_timeout",
                    "skill": skill_name,
                    "controller": controller_name,
                    "timeout_sec": timeout_sec,
                },
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            append_jsonl(
                self.log_path,
                {
                    "timestamp": now_iso(),
                    "event": "action_goal_rejected",
                    "skill": skill_name,
                    "controller": controller_name,
                },
            )
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        if not result_future.done():
            append_jsonl(
                self.log_path,
                {
                    "timestamp": now_iso(),
                    "event": "action_result_timeout",
                    "skill": skill_name,
                    "controller": controller_name,
                    "timeout_sec": timeout_sec,
                },
            )
            return False

        result = result_future.result().result
        error_code = int(result.error_code)
        error_string = str(result.error_string)

        ok = error_code == FollowJointTrajectory.Result.SUCCESSFUL

        append_jsonl(
            self.log_path,
            {
                "timestamp": now_iso(),
                "event": "action_result",
                "skill": skill_name,
                "controller": controller_name,
                "status": "success" if ok else "failed",
                "error_code": error_code,
                "error_string": error_string,
            },
        )

        return ok

    def execute_skill(self, skill: dict) -> bool:
        skill_name = skill.get("name", "")
        params = skill.get("params", {})
        timeout_sec = float(skill.get("timeout_sec", 10.0))
        duration_sec = float(params.get("duration_sec", 3.0))

        append_jsonl(
            self.log_path,
            {
                "timestamp": now_iso(),
                "event": "skill_started",
                "skill": skill_name,
                "params": params,
            },
        )

        t0 = time.time()

        try:
            if skill_name == "move_home":
                positions = params["joint_positions"]
                ok = self.send_trajectory(
                    action_client=self.arm_client,
                    controller_name="arm_controller",
                    joint_names=ARM_JOINTS,
                    positions=positions,
                    duration_sec=duration_sec,
                    timeout_sec=timeout_sec,
                    skill_name=skill_name,
                )

            elif skill_name == "open_gripper":
                positions = params.get("joint_positions", [0.0])
                ok = self.send_trajectory(
                    action_client=self.gripper_client,
                    controller_name="gripper_controller",
                    joint_names=GRIPPER_JOINTS,
                    positions=positions,
                    duration_sec=duration_sec,
                    timeout_sec=timeout_sec,
                    skill_name=skill_name,
                )

            elif skill_name == "close_gripper":
                positions = params.get("joint_positions", [0.6])
                ok = self.send_trajectory(
                    action_client=self.gripper_client,
                    controller_name="gripper_controller",
                    joint_names=GRIPPER_JOINTS,
                    positions=positions,
                    duration_sec=duration_sec,
                    timeout_sec=timeout_sec,
                    skill_name=skill_name,
                )

            else:
                append_jsonl(
                    self.log_path,
                    {
                        "timestamp": now_iso(),
                        "event": "skill_unsupported",
                        "skill": skill_name,
                    },
                )
                ok = False

        except Exception as e:
            append_jsonl(
                self.log_path,
                {
                    "timestamp": now_iso(),
                    "event": "skill_exception",
                    "skill": skill_name,
                    "error": repr(e),
                },
            )
            ok = False

        append_jsonl(
            self.log_path,
            {
                "timestamp": now_iso(),
                "event": "skill_finished",
                "skill": skill_name,
                "status": "success" if ok else "failed",
                "duration_wall_sec": round(time.time() - t0, 3),
            },
        )

        return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-plan", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    task_plan_path = Path(args.task_plan)
    log_path = Path(args.log)

    with task_plan_path.open("r", encoding="utf-8") as f:
        task_plan = json.load(f)

    rclpy.init()
    node = ActionSkillExecutor(log_path)

    append_jsonl(
        log_path,
        {
            "timestamp": now_iso(),
            "event": "executor_started",
            "task_plan": str(task_plan_path),
        },
    )

    try:
        if not node.wait_for_servers():
            append_jsonl(
                log_path,
                {
                    "timestamp": now_iso(),
                    "event": "executor_failed",
                    "reason": "action_server_not_available",
                },
            )
            return 2

        all_ok = True
        for skill in task_plan.get("skills", []):
            ok = node.execute_skill(skill)
            if not ok:
                all_ok = False
                break

        append_jsonl(
            log_path,
            {
                "timestamp": now_iso(),
                "event": "executor_finished",
                "status": "success" if all_ok else "failed",
            },
        )

        return 0 if all_ok else 1

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
