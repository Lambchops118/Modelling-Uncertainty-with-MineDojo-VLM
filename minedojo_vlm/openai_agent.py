"""GPT-6.1 Sol vision-to-action policy using the OpenAI Responses API."""

from __future__ import annotations

from typing import List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from .actions import AgentAction


class ActionDecision(BaseModel):
    observation: str = Field(
        description="A short factual description of relevant visible features."
    )
    rationale: str = Field(
        description="A brief reason the selected action advances the task."
    )
    action: AgentAction
    duration: int = Field(
        ge=1,
        le=100,
        description="Number of controller steps, following the per-action limits.",
    )
    confidence: Literal["low", "medium", "high"]


SYSTEM_INSTRUCTIONS = """You control an embodied agent in Minecraft from a first-person RGB image.
Choose exactly one action from the supplied action schema. Base visual claims only on the image.
Avoid long blind movements: use duration 1-3 near obstacles and at most 8 in clearly open space.
TURN_LEFT and TURN_RIGHT rotate 5 degrees per step and permit 1-18 steps.
LOOK_UP and LOOK_DOWN rotate 5 degrees per step and permit 1-12 steps.
USE interacts with the object under the crosshair. ATTACK performs one strike.
BREAK_BLOCK holds attack continuously for 2-100 steps; use it when mining or chopping a block.
Use duration 1 for USE, ATTACK, and JUMP. WAIT permits 1-6 steps.
This controller cannot click slots in an open chest GUI. To retrieve an item from a chest,
do not USE the chest: target the chest with the crosshair, BREAK_BLOCK long enough to destroy
it, then walk over the dropped contents. The local objective tracker verifies inventory.
Choose STOP only when the objective status explicitly says the target is collected, or when
the task is genuinely impossible or unsafe to continue. Keep the rationale brief."""


class OpenAIVisionAgent:
    def __init__(
        self,
        model: str = "gpt-6.1-sol",
        reasoning_effort: str = "low",
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = client or OpenAI(max_retries=2, timeout=60.0)

    def decide(
        self,
        image_data_url: str,
        task: str,
        history: List[str],
        objective_status: str = "No objective tracker is configured.",
    ):
        history_text = "\n".join(history[-8:]) if history else "No previous actions."
        prompt = (
            f"Task: {task}\n\n"
            f"Objective status: {objective_status}\n\n"
            f"Recent action history:\n{history_text}\n\n"
            "Inspect the current image and select the next action."
        )
        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            instructions=SYSTEM_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": "low",
                        },
                    ],
                }
            ],
            text_format=ActionDecision,
            max_output_tokens=1024,
            store=False,
        )
        if response.status != "completed":
            raise RuntimeError(f"OpenAI response did not complete: {response.status}")
        if response.output_parsed is None:
            raise RuntimeError("OpenAI response contained no parsed action decision")
        return response.output_parsed, response
