# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Priority Panic Environment Client — 15-Step Logic Sync."""

from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Ensure these models match your local models.py
from .models import PriorityPanicAction, PriorityPanicObservation

class PriorityPanicEnv(
    EnvClient[PriorityPanicAction, PriorityPanicObservation, State]
):
    """
    Client for the Priority Panic Environment.
    Handles the communication between your local inference script and the 
    Hugging Face Space or Docker container.
    """

    def _step_payload(self, action: PriorityPanicAction) -> Dict[str, Any]:
        """
        Convert PriorityPanicAction to JSON payload for the server.
        Matches the new 'Clean' server logic for the India AI Hackathon.
        """
        return {
            "ordered_task_ids": action.ordered_task_ids,
            "dropped_task_ids": action.dropped_task_ids,
            "message_to_waiting_person": action.message_to_waiting_person,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PriorityPanicObservation]:
        """
        Parse server response into StepResult[PriorityPanicObservation].
        This fix removes the 'echoed_message' field that was causing the Pydantic crash.
        """
        # Extract the observation block from the server response
        obs_data = payload.get("observation", {})
        
        # Create the Pydantic observation object with the new schema
        observation = PriorityPanicObservation(
            tasks=obs_data.get("tasks", []),
            available_energy=obs_data.get("available_energy", 0),
            waiting_person=obs_data.get("waiting_person", ""),
            level=obs_data.get("level", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object for session tracking.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )