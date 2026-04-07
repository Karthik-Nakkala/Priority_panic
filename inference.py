import asyncio
import json
import os
import textwrap
from typing import Dict, List
from openai import OpenAI
from priority_panic import PriorityPanicAction, PriorityPanicEnv

# --- Configuration ---#

# It will check for API_KEY first (SambaNova), then fallback to HF_TOKEN
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.sambanova.ai/v1"

if not API_KEY:
    print("[ERROR] No API Key found. Please run: $env:API_KEY = 'your_key'")
    exit()

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Crisis Management AI specializing in Task Prioritization. 
    Your goal is to maintain a 'Success Streak' by completing tasks every single step.

    STRATEGY:
    1. PRIORITIZE: High-priority tasks (T1, S3, etc.) cause exponential 'Panic' if left alone. Finish them first.
    2. BUDGET: Each task costs 'energy'. Do not exceed your 'available_energy'.
    3. MOMENTUM: Completing tasks in consecutive steps increases your reward multiplier.
    4. EFFICIENCY: If you have energy left over, try to fit a smaller 'Medium' or 'Low' priority task.

    RESPONSE FORMAT (STRICT JSON):
    {
        "ordered_task_ids": ["ID1", "ID2"],
        "reasoning": "Explain why these tasks were chosen over others."
    }
""").strip()

async def run_level(client: OpenAI, env, level: str) -> float:
    rewards = []
    print(f"[START] task={level} env=priority_panic model={MODEL_NAME}", flush=True)

    result = await env.reset(level=level)
    obs = result.observation
    
    for step in range(1, 16):
        # Edge Case: If no tasks are left, the AI should know it's in 'Monitoring' mode
        task_list = obs.tasks if obs.tasks else "No active tasks. Monitoring for new arrivals."
        
        user_prompt = (
            f"STEP: {step}/15\n"
            f"LEVEL: {obs.level}\n"
            f"ENERGY: {obs.available_energy}\n"
            f"CURRENT TASKS: {task_list}"
        )
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2, # Lower temperature for stable, logical decisions
                response_format={ "type": "json_object" } 
            )
            response_text = completion.choices[0].message.content
            parsed = json.loads(response_text)
        except Exception as e:
            # Edge Case: Handle API Timeouts or Credit exhaustion (402)
            print(f"[DEBUG] API/Parsing Error at step {step}: {e}")
            parsed = {"ordered_task_ids": []}

        # Validate Action (Ensure IDs are strings)
        task_ids = [str(tid) for tid in parsed.get("ordered_task_ids", [])]

        action = PriorityPanicAction(
            ordered_task_ids=task_ids,
            reasoning=parsed.get("reasoning", "Strategic prioritization.")
        )

        result = await env.step(action)
        obs = result.observation
        
        # Reward Tracking
        rewards.append(result.reward)
        
        # Log feedback for the 'training' feel
        status = "WORKING" if task_ids else "IDLE/PANIC"
        print(f"[STEP] {step:02d} | {status} | Action: {task_ids} | Reward: {result.reward:.3f}")
        
        if result.done:
            break

    # Average performance for the level
    final_score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] task={level} score={final_score:.3f}")
    return final_score

async def main():
    if not API_KEY or "hf_" not in API_KEY:
        print("[ERROR] Invalid HF Token. Please check your environment variables.")
        return

    print(f"[DEBUG] Connecting to Space: {HF_SPACE_URL}")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PriorityPanicEnv(base_url=HF_SPACE_URL)

    async with env:
        levels = ["easy", "medium", "hard"]
        total_scores = []
        
        for level in levels:
            score = await run_level(client, env, level)
            total_scores.append(score)
        
        avg_score = sum(total_scores) / len(levels)
        print("\n" + "="*35)
        print(f"FINAL BENCHMARK SCORE: {avg_score:.3f}")
        print("="*35)

if __name__ == "__main__":
    asyncio.run(main())