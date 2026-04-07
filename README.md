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
---

🧩 Problem Statement

AI systems today are highly capable of generating responses but lack the ability to operate in **real-world, dynamic decision-making scenarios**.
There is no standardized, open environment where AI agents can learn to handle **multi-step workflows involving uncertainty, evolving states, and consequences**.

> Inspired by a real situation — two students, four days, no prior experience, and a decision to build anyway.
> Every task in this environment is something we personally faced while building it.
> The agent that learns this, learns what we learned the hard way.

---

💡 Solution

**Priority Panic**, An **Reinforcement Learning (RL)-based OpenEnv environment** where AI agents:

* Interact with **dynamic, stateful scenarios**
* Take **sequential actions** within a defined action space
* Learn through **verifiable reward functions** based on outcomes

This enables **multi-step decision-making, policy learning, and environment simulation**, shifting AI from passive response generation to **active, real-world decision-making systems**.

---

## 🎯 Difficulty Levels

* 🟢 Simple — basic ordering decisions
* 🟡 Medium — multi-step workflows
* 🔴 Complex — multi-constraint planning

---

> This isn’t a toy problem. Every human has lived this.
> Now AI can learn it too.


## Quick Start

The client is **async by default**:
```python
import asyncio
from priority_panic import PriorityPanicAction, PriorityPanicEnv

async def main():
    client = await PriorityPanicEnv.from_docker_image("priority_panic-env:latest")

    async with client:
        result = await client.reset()
        print(f"Level: {result.observation.level}")
        print(f"Tasks: {result.observation.tasks}")

        action = PriorityPanicAction(
            ordered_task_ids=["T1", "T3", "T5", "T2", "T4"],
            dropped_task_ids=[],
            message_to_waiting_person="",
            reasoning="Urgent tasks first"
        )
        result = await client.step(action)
        print(f"Reward: {result.reward}")

asyncio.run(main())
