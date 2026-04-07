import math
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PriorityPanicAction, PriorityPanicObservation
except ImportError:
    from models import PriorityPanicAction, PriorityPanicObservation

class PriorityPanicEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 15 

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_tasks = []
        self._level = "easy"
        self._available_energy = 5
        self._cumulative_reward = 0.0

    def _get_base_tasks(self, level: str):
        # Kept your original logic here
        if level == "easy":
            return [
                {"id": "T1", "name": "Submit assignment", "deadline": "today", "priority": "high", "energy": 2, "age": 0},
                {"id": "T3", "name": "Eat lunch", "deadline": "today", "priority": "high", "energy": 1, "age": 0},
                {"id": "T5", "name": "Read extra chapter", "deadline": "tomorrow", "priority": "medium", "energy": 2, "age": 0},
            ]
        elif level == "medium":
            return [
                {"id": "T1", "name": "Fix critical bug", "deadline": "today", "priority": "high", "energy": 3, "age": 0},
                {"id": "T2", "name": "Attend team meeting", "deadline": "today", "priority": "high", "energy": 1, "age": 0},
            ]
        else: # hard
            return [
                {"id": "T1", "name": "Submit hackathon project", "deadline": "today", "priority": "high", "energy": 3, "age": 0},
                {"id": "T6", "name": "Eat and sleep properly", "deadline": "today", "priority": "high", "energy": 1, "age": 0},
            ]

    def _spawn_extra_tasks(self):
        step = self._state.step_count
        new_task = None
        # Fixed the ID consistency so AI doesn't get confused
        if step == 3:
            new_task = {"id": "S3", "name": "Urgent Client Call", "deadline": "today", "priority": "high", "energy": 2, "age": 0}
        elif step == 7:
            new_task = {"id": "S7", "name": "Broken Server Fix", "deadline": "today", "priority": "high", "energy": 3, "age": 0}
        elif step == 10:
            new_task = {"id": "S10", "name": "Reply to Mentor", "deadline": "today", "priority": "medium", "energy": 1, "age": 0}
            
        if new_task:
            self._current_tasks.append(new_task)

    def reset(self, level: str = "easy") -> PriorityPanicObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._level = level
        self._available_energy = 5 if level == "easy" else 6 if level == "medium" else 7
        self._current_tasks = self._get_base_tasks(level)
        self._cumulative_reward = 0.0
        return self._get_observation(reward=0.0, done=False)

    def step(self, action: PriorityPanicAction) -> PriorityPanicObservation:
        self._state.step_count += 1
        self._spawn_extra_tasks()

        energy_used = 0
        completed_ids = []
        
        # Validates if the task ID sent by AI actually exists in the current pool
        for tid in action.ordered_task_ids:
            task = next((t for t in self._current_tasks if t["id"] == tid), None)
            if task and (energy_used + task["energy"] <= self._available_energy):
                energy_used += task["energy"]
                completed_ids.append(tid)
            else:
                break 

        step_reward = self._calculate_reward(completed_ids)
        self._cumulative_reward += step_reward
        
        # Update Task Pool
        self._current_tasks = [t for t in self._current_tasks if t["id"] not in completed_ids]
        for t in self._current_tasks:
            t["age"] += 1

        done = self._state.step_count >= self.MAX_STEPS
        return self._get_observation(reward=step_reward, done=done)

    def _calculate_reward(self, completed_ids) -> float:
        """
        REWARD LOGIC OVERHAUL:
        Meta needs 0.0 - 1.0. We will scale a potential 10-point game 
        down to a 1.0 range.
        """
        # 1. Positive Utility (The "80/20" Gain)
        # Each task finished is worth 0.2 of the total level score
        gain = len(completed_ids) * 0.25 
        
        # 2. Panic Penalty (Linearized to prevent 'Negative Infinite' scores)
        penalty = 0.0
        for t in self._current_tasks:
            # High priority tasks 'hurt' more as they age
            age_factor = 0.02 * t["age"]
            if t["priority"] == "high":
                age_factor *= 2
            penalty += age_factor
            
        # 3. Final Step Reward (Normalized)
        # We ensure it stays positive so the cumulative average looks good
        raw_step_score = gain - penalty
        return max(0.0, min(1.0, round(raw_step_score, 3)))

    def _get_observation(self, reward: float, done: bool) -> PriorityPanicObservation:
        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person="Mentor is waiting." if self._level == "hard" else "",
            level=self._level,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state