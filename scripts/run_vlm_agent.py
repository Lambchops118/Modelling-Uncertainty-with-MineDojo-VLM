"""Run a bounded GPT-6.1 Sol vision-to-action loop in MineDojo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Allow `python scripts/run_vlm_agent.py` from a source checkout without
# requiring this project itself to be installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import minedojo

from minedojo_vlm.actions import AgentAction, execute_action
from minedojo_vlm.navigation_memory import Area, NavigationMemory
from minedojo_vlm.observations import encode_rgb_as_png_data_url, save_png
from minedojo_vlm.openai_agent import OpenAIVisionAgent


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_minedojo_runtime() -> None:
    """Provide the legacy Java/Conda paths required by MineDojo on this Mac."""
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    java8_home = Path(
        "/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home"
    )
    if sys.platform == "darwin" and java8_home.is_dir():
        os.environ["JAVA_HOME"] = str(java8_home)
        java_bin = str(java8_home / "bin")
        current_path = os.environ.get("PATH", "")
        if not current_path.startswith(java_bin + os.pathsep):
            os.environ["PATH"] = java_bin + os.pathsep + current_path


def serialize_usage(response) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {"repr": repr(usage)}


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def inventory_quantity(observation, target_item: str) -> int:
    """Return the quantity of a named item in MineDojo's flat inventory."""
    inventory = observation.get("inventory", {})
    names = inventory.get("name", [])
    quantities = inventory.get("quantity", [])
    normalized_target = target_item.replace("minecraft:", "").replace("_", " ")
    total = 0
    for name, quantity in zip(names, quantities):
        normalized_name = str(name).replace("minecraft:", "").replace("_", " ")
        if normalized_name == normalized_target:
            total += int(quantity)
    return total


def inventory_status(observation) -> str:
    """Describe non-empty inventory and the selected first hotbar slot."""
    inventory = observation.get("inventory", {})
    names = list(inventory.get("name", []))
    quantities = list(inventory.get("quantity", []))
    items = []
    for name, quantity in zip(names, quantities):
        normalized_name = str(name).replace("minecraft:", "").replace("_", " ")
        quantity = int(quantity)
        if normalized_name != "air" and quantity > 0:
            items.append(f"{normalized_name} x{quantity}")
    held_item = "empty hand"
    equipment = observation.get("equipment", {})
    equipment_names = list(equipment.get("name", []))
    equipment_quantities = list(equipment.get("quantity", []))
    if equipment_names:
        mainhand = str(equipment_names[0]).replace("minecraft:", "").replace("_", " ")
        if mainhand != "air" and int(equipment_quantities[0]) > 0:
            held_item = mainhand
    elif names:
        # Older MineDojo builds may omit the equipment observation. Slot zero is
        # the best available approximation in that case.
        slot_zero = str(names[0]).replace("minecraft:", "").replace("_", " ")
        if slot_zero != "air" and int(quantities[0]) > 0:
            held_item = slot_zero
    contents = ", ".join(items) if items else "empty"
    return (
        f"Held item rendered at lower-right: {held_item}. "
        f"Non-empty inventory: {contents}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default=(
            "Search the three colored-floor rooms for the chest containing a diamond. "
            "Break chests to spill their contents, collect the diamond by walking over it, "
            "and stop only after the objective tracker confirms that the diamond is in inventory."
        ),
    )
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to runs/<UTC timestamp>.",
    )
    parser.add_argument(
        "--action-step-delay",
        type=float,
        default=0.05,
        help="Wall-clock pacing between controller steps (default: 0.05s / 20Hz).",
    )
    parser.add_argument(
        "--world-path",
        type=Path,
        default=Path(
            "/Users/jacksal1/Library/Application Support/minecraft/saves/"
            "Enrichment Center 1"
        ),
        help="Minecraft 1.11.2 saved-world directory containing level.dat.",
    )
    parser.add_argument("--spawn-x", type=float, default=365.0)
    parser.add_argument("--spawn-y", type=float, default=57.0)
    parser.add_argument("--spawn-z", type=float, default=-477.0)
    parser.add_argument(
        "--spawn-yaw",
        type=float,
        default=-90.0,
        help="Minecraft yaw in degrees; -90 faces east.",
    )
    parser.add_argument("--spawn-pitch", type=float, default=0.0)
    parser.add_argument("--target-item", default="diamond")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_minedojo_runtime()
    if args.max_decisions < 1:
        raise SystemExit("--max-decisions must be at least 1")
    world_path = args.world_path.expanduser().resolve()
    if not (world_path / "level.dat").is_file():
        raise SystemExit(
            f"--world-path must be a saved-world directory containing level.dat: {world_path}"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it in this terminal before running."
        )

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or Path("runs") / run_name
    frames_dir = run_dir / "frames"
    events_path = run_dir / "events.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "created_at": utc_timestamp(),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "task": args.task,
                "max_decisions": args.max_decisions,
                "mine_dojo_task": "open-ended",
                "world_path": str(world_path),
                "start_position": {
                    "x": args.spawn_x,
                    "y": args.spawn_y,
                    "z": args.spawn_z,
                    "yaw": args.spawn_yaw,
                    "pitch": args.spawn_pitch,
                },
                "target_item": args.target_item,
                "image_size": [360, 640],
                "camera_interval_degrees": 5,
                "action_step_delay": args.action_step_delay,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    agent = OpenAIVisionAgent(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    history = []
    action_history = []
    navigation_memory = NavigationMemory()
    pending_break_room = None
    env = minedojo.make(
        task_id="open-ended",
        image_size=(360, 640),
        cam_interval=5,
        generate_world_type="from_file",
        world_file_path=str(world_path),
        start_position={
            "x": args.spawn_x,
            "y": args.spawn_y,
            "z": args.spawn_z,
            "yaw": args.spawn_yaw,
            "pitch": args.spawn_pitch,
        },
        allow_time_passage=False,
        allow_mob_spawn=False,
        initial_weather="clear",
    )

    print(f"[INFO] Run data: {run_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Task: {args.task}")
    try:
        observation = env.reset()
        target_quantity = inventory_quantity(observation, args.target_item)
        if target_quantity > 0:
            raise RuntimeError(
                f"Agent started with {target_quantity} {args.target_item}; "
                "the objective would already be complete"
            )
        termination_reason = "decision_budget_exhausted"
        for decision_index in range(args.max_decisions):
            image_data_url, png_bytes = encode_rgb_as_png_data_url(
                observation["rgb"]
            )
            frame_path = frames_dir / f"frame_{decision_index:03d}.png"
            save_png(png_bytes, frame_path)

            started = time.monotonic()
            objective_status = (
                f"Inventory contains {target_quantity} {args.target_item}. "
                f"Success requires at least 1. {inventory_status(observation)}"
            )
            if pending_break_room is not None:
                objective_status += (
                    f" Verification is pending for an attempted "
                    f"{pending_break_room.value} chest break. Inspect whether that "
                    "chest block is visibly gone and report previous_break_succeeded."
                )
            else:
                objective_status += " No chest break is awaiting verification."
            memory_text = navigation_memory.prompt_text()
            recent_turns = action_history[-3:]
            if len(recent_turns) == 3 and all(
                action in {AgentAction.TURN_LEFT, AgentAction.TURN_RIGHT}
                for action in recent_turns
            ):
                memory_text += (
                    "\n- LOOP_ALERT: three consecutive turn actions made no navigational "
                    "progress. Do not chase an object at the lower-right edge; it is the "
                    "held item. Look toward the center of the scene or leave this room."
                )
            decision, response = agent.decide(
                image_data_url,
                args.task,
                history,
                objective_status=objective_status,
                navigation_memory=memory_text,
            )
            if pending_break_room is not None:
                if decision.previous_break_succeeded:
                    navigation_memory.mark_chest_checked(pending_break_room)
                    print(f"[MEMORY] Confirmed {pending_break_room.value} chest destroyed")
                else:
                    print(
                        f"[MEMORY] {pending_break_room.value} chest destruction was not "
                        "confirmed; room remains unchecked"
                    )
                pending_break_room = None
            navigation_memory.observe(decision.current_area)
            action_history.append(decision.action)
            latency = time.monotonic() - started
            print(
                f"[DECISION {decision_index:02d}] {decision.action.value} "
                f"x{decision.duration} ({decision.confidence} confidence)"
            )
            print(f"  observed: {decision.observation}")
            print(f"  rationale: {decision.rationale}")

            record = {
                "event": "decision",
                "timestamp": utc_timestamp(),
                "decision_index": decision_index,
                "frame": str(frame_path),
                "task": args.task,
                "model": args.model,
                "response_id": response.id,
                "latency_seconds": round(latency, 6),
                "decision": decision.model_dump(mode="json"),
                "objective_status": objective_status,
                "navigation_memory": navigation_memory.model_dump(mode="json"),
                "usage": serialize_usage(response),
            }
            append_jsonl(events_path, record)

            if decision.action == AgentAction.STOP:
                print("[INFO] Model selected STOP")
                termination_reason = "model_stop"
                break

            executed_action = decision.action
            executed_duration = decision.duration
            if decision.action == AgentAction.BREAK_BLOCK and decision.crosshair_on_chest:
                if decision.target_chest_room == Area.UNKNOWN:
                    executed_action = AgentAction.WAIT
                    executed_duration = 1
                    print("[GUARD] Chest room is unknown; refusing untracked break")
                elif decision.current_area != decision.target_chest_room:
                    executed_action = AgentAction.MOVE_FORWARD
                    executed_duration = 6
                    print(
                        f"[GUARD] Chest is in {decision.target_chest_room.value} but agent "
                        f"is in {decision.current_area.value}; approaching before breaking"
                    )
                else:
                    executed_duration = 100
                    pending_break_room = decision.target_chest_room

            history.append(
                f"Step {decision_index}: area={decision.current_area.value}; "
                f"saw {decision.observation}; requested {decision.action.value} "
                f"x{decision.duration}; executed {executed_action.value} "
                f"x{executed_duration}."
            )

            transition = execute_action(
                env,
                executed_action,
                executed_duration,
                step_delay=args.action_step_delay,
            )
            if transition is None:
                raise RuntimeError("MineDojo returned no transition")
            observation, reward, done, _ = transition
            target_quantity = inventory_quantity(observation, args.target_item)
            success = target_quantity >= 1
            append_jsonl(
                events_path,
                {
                    "event": "action_result",
                    "timestamp": utc_timestamp(),
                    "decision_index": decision_index,
                    "reward": float(reward),
                    "done": bool(done),
                    "target_item": args.target_item,
                    "target_quantity": target_quantity,
                    "success": success,
                    "requested_action": decision.action.value,
                    "requested_duration": decision.duration,
                    "executed_action": executed_action.value,
                    "executed_duration": executed_duration,
                    "pending_break_room": (
                        pending_break_room.value
                        if pending_break_room is not None
                        else None
                    ),
                    "navigation_memory": navigation_memory.model_dump(mode="json"),
                },
            )
            if success:
                print(
                    f"[SUCCESS] Collected {target_quantity} {args.target_item}; "
                    "objective verified from inventory"
                )
                termination_reason = "success"
                break
            if done:
                print("[INFO] MineDojo episode ended")
                termination_reason = "environment_done"
                break
        append_jsonl(
            events_path,
            {
                "event": "termination",
                "timestamp": utc_timestamp(),
                "reason": termination_reason,
                "target_item": args.target_item,
                "target_quantity": target_quantity,
                "navigation_memory": navigation_memory.model_dump(mode="json"),
            },
        )
    finally:
        env.close()
        print("[INFO] MineDojo closed cleanly")


if __name__ == "__main__":
    main()
