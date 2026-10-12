"""Launch MineDojo and demonstrate several basic movement actions."""

from __future__ import annotations

import argparse
import time

import minedojo


def perform(env, label: str, count: int, configure, delay: float) -> bool:
    """Perform one named action repeatedly; return True if the episode ends."""
    print(f"[ACTION] {label} ({count} steps)")
    for _ in range(count):
        action = env.action_space.no_op()
        configure(action)
        _, reward, done, _ = env.step(action)
        if reward:
            print(f"[INFO] Reward: {reward}")
        if done:
            print("[INFO] Episode ended early")
            return True
        if delay:
            time.sleep(delay)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.10,
        help="Pause in seconds after each step so movement is easy to see.",
    )
    args = parser.parse_args()

    print("[INFO] Creating MineDojo environment")
    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(360, 640),
    )

    try:
        observation = env.reset()
        print(f"[INFO] Reset succeeded; RGB shape: {observation['rgb'].shape}")
        print(f"[INFO] Action vector: {env.action_space.no_op()}")
        time.sleep(2)

        demonstrations = [
            ("walk forward", 20, lambda action: action.__setitem__(0, 1)),
            # Camera actions are relative. With MineDojo's default 15-degree
            # camera interval, bin 13 is a 15-degree right turn per step.
            ("turn camera right 90 degrees", 6, lambda action: action.__setitem__(4, 13)),
            (
                "walk forward and jump",
                12,
                lambda action: (
                    action.__setitem__(0, 1),
                    action.__setitem__(2, 1),
                ),
            ),
            ("strafe left", 10, lambda action: action.__setitem__(1, 1)),
            ("walk backward", 10, lambda action: action.__setitem__(0, 2)),
        ]

        for label, count, configure in demonstrations:
            if perform(env, label, count, configure, args.delay):
                break

        print("[INFO] Demonstration complete; leaving the window open for 3 seconds")
        time.sleep(3)
    finally:
        env.close()
        print("[INFO] MineDojo closed cleanly")


if __name__ == "__main__":
    main()
