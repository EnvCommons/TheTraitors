"""
Test agent for The Traitors environment.
Runs a single task end-to-end using the OpenReward SDK with rollout logging.

Usage:
    source /Users/rosstaylor/Documents/or_envs/newenvs/.env
    python server.py &
    python test_agent.py
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ENV_NAME = "local/TheTraitors"
SPLIT = "train"


async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(
        name=ENV_NAME, base_url="http://localhost:8080"
    )
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Tools ({len(tools)}): {tools[0] if tools else 'none'}")

    task = tasks[0]
    print(f"\nRunning task: {task.task_spec}")

    await run_task(or_client, environment, oai_client, task, tools)


async def run_task(or_client, environment, oai_client, task, tools) -> None:
    """Run a single task with the agent and log to OpenReward rollout."""
    finished = False
    total_reward = 0.0
    max_turns = 200

    # Create rollout for this task
    rollout = or_client.rollout.create(
        run_name="thetraitors_test",
        rollout_name="test_run",
        environment=ENV_NAME,
        split=SPLIT,
        task_spec=task.task_spec,
    )

    async with environment.session(
        task=task, secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text if hasattr(prompt[0], "text") else str(prompt[0])
        input_list = [{"role": "user", "content": prompt_text}]

        print(f"\n{'='*60}")
        print("GAME STARTED")
        print(f"{'='*60}")
        print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)

        # Log initial user prompt
        rollout.log_openai_response(message=input_list[0], is_finished=False)

        turn = 0
        while not finished and turn < max_turns:
            turn += 1
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Log model response
            rollout.log_openai_response(response.output[-1])

            # Process response
            tool_called = False
            for item in response.output:
                input_list.append(item.model_dump())

                if item.type == "function_call":
                    tool_called = True
                    args = json.loads(str(item.arguments))
                    print(f"\n--- Turn {turn} ---")
                    print(f"Tool: {item.name}({json.dumps(args, indent=2)[:200]})")

                    tool_result = await session.call_tool(item.name, args)

                    reward = tool_result.reward
                    finished = tool_result.finished
                    total_reward += reward if reward else 0.0

                    result_text = (
                        tool_result.blocks[0].text if tool_result.blocks else ""
                    )
                    print(f"Result: {result_text[:300]}...")
                    print(f"Reward: {reward:.3f} | Total: {total_reward:.3f}")

                    tool_output = {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_text,
                    }
                    input_list.append(tool_output)

                    # Log tool output with reward and finished status
                    rollout.log_openai_response(
                        tool_output,
                        reward=reward,
                        is_finished=finished,
                    )

                    if finished:
                        print(f"\n{'='*60}")
                        print("GAME FINISHED!")
                        print(f"Final reward: {total_reward:.3f}")
                        print(f"{'='*60}")
                        break

                elif item.type == "text":
                    print(f"\nModel text: {item.text[:200]}...")

            if not tool_called:
                print("\nNo tool call in response, ending task")
                break

        print(f"\nTask completed in {turn} turns")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
