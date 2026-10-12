"""Persistent, compact memory for the colored-room smoke test."""

from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field


class Area(str, Enum):
    RED_ROOM = "RED_ROOM"
    BLUE_ROOM = "BLUE_ROOM"
    YELLOW_ROOM = "YELLOW_ROOM"
    CENTRAL_ROOM = "CENTRAL_ROOM"
    CORRIDOR = "CORRIDOR"
    UNKNOWN = "UNKNOWN"


COLORED_ROOMS = (Area.RED_ROOM, Area.BLUE_ROOM, Area.YELLOW_ROOM)


class RoomState(BaseModel):
    visited: bool = False
    chest_checked: bool = False


class NavigationMemory(BaseModel):
    rooms: Dict[Area, RoomState] = Field(
        default_factory=lambda: {room: RoomState() for room in COLORED_ROOMS}
    )
    last_colored_room: Area = Area.UNKNOWN

    def observe(self, area: Area) -> None:
        if area in COLORED_ROOMS:
            self.rooms[area].visited = True
            self.last_colored_room = area

    def mark_chest_checked(self, area: Area) -> None:
        if area in COLORED_ROOMS:
            self.observe(area)
            self.rooms[area].chest_checked = True

    def prompt_text(self) -> str:
        lines = []
        for room in COLORED_ROOMS:
            state = self.rooms[room]
            lines.append(
                f"- {room.value}: visited={str(state.visited).lower()}, "
                f"chest_checked={str(state.chest_checked).lower()}"
            )
        unchecked = [
            room.value for room in COLORED_ROOMS if not self.rooms[room].chest_checked
        ]
        lines.append(
            "- unchecked_rooms: " + (", ".join(unchecked) if unchecked else "none")
        )
        lines.append(f"- last_colored_room: {self.last_colored_room.value}")
        return "\n".join(lines)
