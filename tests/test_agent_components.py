from __future__ import annotations

import unittest

import numpy as np

from minedojo_vlm.actions import AgentAction, execute_action
from minedojo_vlm.observations import encode_rgb_as_png_data_url, rgb_to_image
from minedojo_vlm.openai_agent import ActionDecision
from minedojo_vlm.navigation_memory import Area, NavigationMemory


class FakeActionSpace:
    def no_op(self):
        return np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int64)


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.actions = []

    def step(self, action):
        self.actions.append(action.copy())
        return {"rgb": np.zeros((3, 4, 5), dtype=np.uint8)}, 0.0, False, {}


class AgentComponentTests(unittest.TestCase):
    def test_chw_image_conversion_and_encoding(self):
        rgb = np.zeros((3, 4, 5), dtype=np.uint8)
        image = rgb_to_image(rgb)
        self.assertEqual(image.size, (5, 4))
        data_url, png_bytes = encode_rgb_as_png_data_url(rgb)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertTrue(png_bytes.startswith(b"\x89PNG"))

    def test_action_schema_rejects_excessive_duration(self):
        with self.assertRaises(ValueError):
            ActionDecision(
                observation="Open ground ahead.",
                rationale="Move forward.",
                current_area=Area.CORRIDOR,
                crosshair_on_chest=False,
                target_chest_room=Area.UNKNOWN,
                previous_break_succeeded=False,
                action=AgentAction.MOVE_FORWARD,
                duration=101,
                confidence="high",
            )

    def test_navigation_memory_persists_checked_rooms(self):
        memory = NavigationMemory()
        memory.observe(Area.RED_ROOM)
        memory.mark_chest_checked(Area.RED_ROOM)
        memory.observe(Area.BLUE_ROOM)

        self.assertTrue(memory.rooms[Area.RED_ROOM].chest_checked)
        self.assertTrue(memory.rooms[Area.BLUE_ROOM].visited)
        self.assertIn("YELLOW_ROOM", memory.prompt_text())

    def test_turn_right_translation(self):
        env = FakeEnv()
        execute_action(env, AgentAction.TURN_RIGHT, 2, step_delay=0)
        self.assertEqual(len(env.actions), 3)
        self.assertTrue(all(action[4] == 13 for action in env.actions[:2]))
        self.assertEqual(env.actions[-1][4], 12)

    def test_break_block_holds_then_releases_attack(self):
        env = FakeEnv()
        execute_action(env, AgentAction.BREAK_BLOCK, 4, step_delay=0)
        self.assertTrue(all(action[5] == 3 for action in env.actions[:4]))
        self.assertEqual(env.actions[-1][5], 0)


if __name__ == "__main__":
    unittest.main()
