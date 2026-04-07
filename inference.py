"""
Inference Script — Priority Panic Environment (Multi-Step Overhaul)
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI
from priority_panic import PriorityPanicAction, PriorityPanicEnv

# --- Configuration & Environment Variables ---
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_SPACE_URL = os.getenv("HF_SPACE_URL") or "https://madhubuilds-priority-panic.hf.space"

# Updated to specifically check for your 'HF.Token' name
API_KEY = (
    os.getenv("HF.Token") or 
    os.getenv("HF_TOKEN") or 
    "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"
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# --- Benchmark Hyperparameters ---
BENCHMARK = "priority_panic"
MAX_STEPS = 15 
TEMPERATURE = 0.4 
MAX_TOKENS = 500

def validate_config() -> bool:
    """Validate required environment variables before execution."""
    errors = []
    if not API_KEY:
        errors.append("ERROR: HF_TOKEN / HF.Token not set.")
    if not IMAGE_NAME and not HF_SPACE_URL:
        errors.append("ERROR: Neither LOCAL_IMAGE_NAME nor HF_SPACE_URL set")
    
    if errors:
        print("[START] task=validation status=failed", flush=True)
        for err in errors:
            print(f"[DEBUG] {err}", flush=True)
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        return False
    return True

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Task Prioritization Agent. Your goal is to maximize cumulative reward by completing tasks before they "Panic."
    
    RULES:
    1. Binary Completion: A task is only finished if you have enough energy for it in the CURRENT step.
    2. Energy: You have a limited energy bandwidth per step. 
    3. Panic: Tasks age every step they are not finished. Older tasks carry exponential negative penalties.
    4. Respond ONLY in JSON format:
    {
        "ordered_task_ids": ["T1", "T2"],
        "dropped_task_ids": ["T5"],
        "message_to_waiting_person": "string",
        "reasoning": "string"
    }
""").strip()

def log_step(step: int, action: list, reward: float, done: bool) -> None:
    done_val = str(done).lower()
    # Log the step with the normalized reward
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done_val} error=null", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, observation: Dict) -> Dict:
    """Request prioritization decision from the LLM."""
    user_prompt = f"Level: {observation['level']} | Energy: {observation['available_energy']}\nTasks: {observation['tasks']}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={ "type": "json_object" } 
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as exc:
        # If we see a 401 here, it confirms the token is invalid
        print(f"[DEBUG] API Error: {exc}", flush=True)
        return {"ordered_task_ids": [], "dropped_task_ids": [], "reasoning": "Error fallback"}

async def run_level(client: OpenAI, env, level: str) -> float:
    rewards = []
    print(f"[START] task={level} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    result = await env.reset(level=level)
    obs = result.observation
    
    total_steps = 0
    for step in range(1, MAX_STEPS + 1):
        total_steps = step
        
        parsed = get_model_action(client, {
            "level": obs.level,
            "available_energy": obs.available_energy,
            "tasks": obs.tasks
        })

        action = PriorityPanicAction(
            ordered_task_ids=parsed.get("ordered_task_ids", []),
            dropped_task_ids=parsed.get("dropped_task_ids", []),
            message_to_waiting_person=parsed.get("message_to_waiting_person", ""),
            reasoning=parsed.get("reasoning", "")
        )

        result = await env.step(action)
        obs = result.observation
        
        # NORMALIZATION: Ensure reward is 0.0-1.0 for the rubric
        normalized_step_reward = max(0.0, min(1.0, float(result.reward)))
        rewards.append(normalized_step_reward)
        
        log_step(step, parsed.get("ordered_task_ids", []), normalized_step_reward, result.done)
        
        if result.done:
            break

    # Final level score is the average of normalized rewards
    final_score = sum(rewards) / len(rewards) if rewards else 0.0
    success = final_score > 0.1
    log_end(success, total_steps, final_score, rewards)
    return final_score

async def main():
    if not validate_config():
        return

    print(f"[DEBUG] Connecting to: {HF_SPACE_URL}", flush=True)
    # The API_KEY here is now pulling from your HF.Token
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    if IMAGE_NAME:
        env = await PriorityPanicEnv.from_docker_image(IMAGE_NAME)
    else:
        env = PriorityPanicEnv(base_url=HF_SPACE_URL)

    async with env:
        levels = ["easy", "medium", "hard"]
        scores = []
        for level in levels:
            score = await run_level(client, env, level)
            scores.append(score)
        
        final_avg = sum(scores) / len(levels)
        
        print("-" * 30)
        print(f"[DEBUG] Final Average Score: {final_avg:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())