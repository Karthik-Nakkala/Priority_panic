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
    MAX_STEPS = 15  # Fixed 15-step horizon for multi-step RL

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_tasks = []
        self._level = "easy"
        self._available_energy = 5
        self._cumulative_reward = 0.0

    def _get_base_tasks(self, level: str):
        """Initial task loadout at step 0."""
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
        """Injects new tasks at specific step intervals."""
        step = self._state.step_count
        new_task = None
        
        if step == 3:
            new_task = {"id": f"S3", "name": "Urgent Client Call", "deadline": "today", "priority": "high", "energy": 2, "age": 0}
        elif step == 7:
            new_task = {"id": f"S7", "name": "Broken Server Fix", "deadline": "today", "priority": "high", "energy": 3, "age": 0}
        elif step == 10:
            new_task = {"id": f"S10", "name": "Reply to Mentor", "deadline": "today", "priority": "medium", "energy": 1, "age": 0}
            
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
        
        # 1. Spawn new tasks
        self._spawn_extra_tasks()

        # 2. Process Actions (Binary Completion Logic)
        energy_used = 0
        completed_ids = []
        for tid in action.ordered_task_ids:
            task = next((t for t in self._current_tasks if t["id"] == tid), None)
            if task and (energy_used + task["energy"] <= self._available_energy):
                energy_used += task["energy"]
                completed_ids.append(tid)
            else:
                break # Stop if energy exhausted

        # 3. Calculate Reward & Update State
        step_reward = self._calculate_reward(completed_ids, energy_used)
        self._cumulative_reward += step_reward
        
        # Remove completed tasks
        self._current_tasks = [t for t in self._current_tasks if t["id"] not in completed_ids]
        
        # Age remaining tasks
        for t in self._current_tasks:
            t["age"] += 1

        # 4. Check Termination
        done = self._state.step_count >= self.MAX_STEPS
        
        return self._get_observation(reward=step_reward, done=done)

    def _calculate_reward(self, completed_ids, energy_used) -> float:
        reward = len(completed_ids) * 0.2 # Base reward for finishing tasks
        
        # Exponential Panic Penalty for uncompleted tasks
        panic_penalty = 0.0
        multiplier = 0.2 if self._level == "hard" else 0.1
        
        for t in self._current_tasks:
            # Penalty increases as age increases
            penalty = 0.05 * math.exp(t["age"] * multiplier)
            if t["priority"] == "high":
                penalty *= 2
            panic_penalty += penalty
            
        return round(reward - panic_penalty, 3)

    def _get_observation(self, reward: float, done: bool) -> PriorityPanicObservation:
        waiting = ""
        if self._level == "hard" and self._state.step_count == 0:
            waiting = "Your mentor Priya is waiting for a project update."

        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person=waiting,
            level=self._level,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state