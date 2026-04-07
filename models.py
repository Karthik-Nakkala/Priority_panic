# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Priority Panic Environment Models — India AI Hackathon '26 Edition."""

from typing import List, Dict, Optional
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation

class PriorityPanicObservation(Observation):
    """What the agent sees — a pile of tasks and constraints."""
    
    tasks: List[Dict] = Field(
        default_factory=list, 
        description="List of tasks to prioritize (id, name, energy, deadline)"
    )
    available_energy: int = Field(
        default=5, 
        description="Total energy the agent has for this step"
    )
    waiting_person: str = Field(
        default="", 
        description="Name of the person waiting for a status update"
    )
    level: str = Field(
        default="easy", 
        description="Difficulty: easy (5 steps), medium (10 steps), or hard (15 steps)"
    )
    
    # Required for OpenEnv compatibility
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Optional[Dict] = Field(default_factory=dict)

    # This prevents the "Extra inputs are not permitted" crash
    model_config = ConfigDict(extra='ignore')


class PriorityPanicAction(Action):
    """What the agent decides to do in response to the pressure."""
    
    ordered_task_ids: List[str] = Field(
        default_factory=list, 
        description="Task IDs in priority order to be processed"
    )
    dropped_task_ids: List[str] = Field(
        default_factory=list, 
        description="Tasks the agent explicitly chooses to abandon"
    )
    message_to_waiting_person: str = Field(
        default="", 
        description="Optional status update message"
    )
    reasoning: str = Field(
        default="", 
        description="The agent's internal logic for the hackathon judges"
    )

    model_config = ConfigDict(extra='ignore')