---
title: Priority Panic
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: other
short_description: 'RL environment: prioritize tasks under pressure.'
tags:
  - openenv
  - reinforcement-learning
  - pytorch
---

# 🚨 Priority Panic: OpenEnv Decision-Making Environment

AI systems today are highly capable of generating responses but often struggle in **real-world, dynamic decision-making scenarios**. **Priority Panic** is a Reinforcement Learning (RL) environment designed to bridge this gap, forcing agents to operate within evolving states and mounting consequences.

> **The Backstory:** Inspired by a real situation — two students, few days left for submission, no prior experience, and a decision to build anyway. Every task in this environment is something we personally faced while building it. The agent that learns this, learns what we learned the hard way.
> **"This isn’t a toy problem. Every human has lived this. Now AI can learn it too."**
---

## 🧠 The "Panic" Mechanism (Core Innovation)

This environment isn't just about ordering a list; it’s about surviving a **"Panic Spiral."**

* **Exponential Age Penalty:** For every step a task remains unfinished, its negative weight grows exponentially. 
    * *Formula:* $Penalty = -0.1 \times e^{(0.2 \times \text{task\_age})}$
* **Dynamic Pressure:** The environment is alive. New "Panic Tasks" spawn unexpectedly at **Steps 3, 7, and 10**, simulating real-world interruptions and deadline shifts.
* **Binary Completion:** Tasks are only "cleared" if the agent allocates sufficient energy bandwidth in a single step, requiring strategic resource management.

---

## 📊 Benchmark Results (Proven Baseline)

We benchmarked this environment using the **Qwen/Qwen2.5-72B-Instruct** model over a full **15-step horizon**. The results demonstrate the environment's ability to challenge high-tier LLMs.

| Difficulty | Steps | Cumulative Score | Status |
| :--- | :--- | :--- | :--- |
| 🟢 **Easy** | 15 | 2.689 | ✅ Success |
| 🟡 **Medium** | 15 | 8.534 | ✅ Success |
| 🔴 **Hard** | 15 | 17.190 | ✅ Success |
| **OVERALL** | **15** | **9.471** | **Verified Baseline** |

---

## 💡 Solution Architecture

We introduce a **stateful OpenEnv environment** where AI agents:
* Interact with **dynamic scenarios** (Tasks, Energy, Waiting Persons).
* Take **sequential actions** (Prioritizing vs. Dropping vs. Communicating).
* Learn through **verifiable reward functions** that penalize procrastination and reward efficiency.

---

## 🎯 Difficulty Levels

* **Simple:** Basic ordering decisions with ample energy.
* **Medium:** Multi-step workflows with task dependencies.
* **Complex:** Multi-constraint planning with dynamic spawns and low energy.

---

## 🚀 Quick Start

The client is **async by default** and compatible with the OpenEnv framework.

```python
import asyncio
from priority_panic import PriorityPanicAction, PriorityPanicEnv

async def main():
    # Connect to the live Hugging Face Space
    client = PriorityPanicEnv(base_url="[https://madhubuilds-priority-panic.hf.space](https://madhubuilds-priority-panic.hf.space)")

    async with client:
        result = await client.reset(level="hard")
        print(f"Current Tasks: {result.observation.tasks}")

        action = PriorityPanicAction(
            ordered_task_ids=["T1", "T3"],
            dropped_task_ids=["T5"],
            message_to_waiting_person="I'm on it!",
            reasoning="T1 is aging rapidly; T3 is high priority."
        )
        
        result = await client.step(action)
        print(f"Step Reward: {result.reward}")

asyncio.run(main())