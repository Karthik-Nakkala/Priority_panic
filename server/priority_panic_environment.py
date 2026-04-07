import math
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
        self._streak = 0  # New: Tracks consecutive successful steps
        self._cumulative_reward = 0.0

    def _get_base_tasks(self, level: str):
        loadouts = {
            "easy": [
                {"id": "T1", "name": "Submit assignment", "priority": "high", "energy": 2, "age": 0},
                {"id": "T3", "name": "Eat lunch", "priority": "high", "energy": 1, "age": 0},
                {"id": "T5", "name": "Read extra chapter", "priority": "medium", "energy": 2, "age": 0},
            ],
            "medium": [
                {"id": "T1", "name": "Fix critical bug", "priority": "high", "energy": 3, "age": 0},
                {"id": "T2", "name": "Attend team meeting", "priority": "high", "energy": 1, "age": 0},
                {"id": "T8", "name": "Update Documentation", "priority": "low", "energy": 2, "age": 0},
            ],
            "hard": [
                {"id": "T1", "name": "Hackathon Final Push", "priority": "high", "energy": 4, "age": 0},
                {"id": "T6", "name": "Server Migration", "priority": "high", "energy": 3, "age": 0},
            ]
        }
        return loadouts.get(level, loadouts["easy"])

    def step(self, action: PriorityPanicAction) -> PriorityPanicObservation:
        self._state.step_count += 1
        
        # Edge Case: Handle empty or null action
        ids_to_process = action.ordered_task_ids if action.ordered_task_ids else []
        
        energy_used = 0
        completed_ids = []
        
        # Process Actions with Energy Validation
        for tid in ids_to_process:
            task = next((t for t in self._current_tasks if t["id"] == tid), None)
            if task:
                if energy_used + task["energy"] <= self._available_energy:
                    energy_used += task["energy"]
                    completed_ids.append(tid)
                # Edge Case: If energy exceeded, we skip this task but continue 
                # checking others (in case a smaller task fits).

        # Reward Logic with Momentum
        step_reward = self._calculate_reward(completed_ids)
        self._cumulative_reward += step_reward
        
        # Update Streak
        if len(completed_ids) > 0:
            self._streak += 1
        else:
            self._streak = 0 # Reset streak if no work done

        # Remove completed and update age
        self._current_tasks = [t for t in self._current_tasks if t["id"] not in completed_ids]
        for t in self._current_tasks:
            t["age"] += 1

        # Periodic Task Injection (Keeps the AI busy)
        if self._state.step_count in [3, 7, 12]:
            self._current_tasks.append({
                "id": f"S{self._state.step_count}", 
                "name": "Incoming Request", 
                "priority": "high" if self._level != "easy" else "medium",
                "energy": 2, "age": 0
            })

        done = self._state.step_count >= self.MAX_STEPS
        return self._get_observation(reward=step_reward, done=done)

    def _calculate_reward(self, completed_ids) -> float:
        """
        ADVANCED REWARD SHAPING:
        1. Accomplishment (Base): 0.3 per task.
        2. Momentum Bonus: Adds 10% extra reward for every step in a streak.
        3. Panic Penalty: Linear (not exponential) so AI can still recover.
        """
        if not completed_ids and not self._current_tasks:
            return 0.1 # Idle reward if everything is done (Peace of mind)

        # 1. Base Utility
        base_gain = len(completed_ids) * 0.3
        
        # 2. Streak/Momentum Multiplier (Max 1.5x)
        multiplier = min(1.5, 1.0 + (self._streak * 0.1))
        adjusted_gain = base_gain * multiplier
        
        # 3. Penalty Logic (Edge Case: Penalty shouldn't exceed Gain)
        penalty = 0.0
        for t in self._current_tasks:
            # High priority aging is the main threat
            p = 0.05 + (0.02 * t["age"])
            if t["priority"] == "high": p *= 2
            penalty += p

        # 4. Final Normalization for Meta (0.0 - 1.0)
        # We divide by 10 to ensure 9.1 raw becomes 0.91
        raw_score = (adjusted_gain - penalty)
        return max(0.0, min(1.0, (raw_score + 2.0) / 10.0)) # Offset by +2 to avoid 0.0 early on

    def _get_observation(self, reward: float, done: bool) -> PriorityPanicObservation:
        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            level=self._level,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State: return self._state
    def reset(self, level: str = "easy"):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._level = level
        self._available_energy = 5 if level == "easy" else 6
        self._current_tasks = self._get_base_tasks(level)
        self._streak = 0
        return self._get_observation(reward=0.0, done=False)