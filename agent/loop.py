import json
import re
from config import DEBUG, SYSTEM_PROMPT
from agent.registry import TOOLS, TOOL_DESCRIPTIONS, tool_help
from core.model import generate_response


def parse_tool_call(response):
    """
    Check if the model's response contains a tool call
    Returns (tool_name, args_dict) or (None, None) if no tool call found
    """
    tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL)
    args_match = re.search(r'<args>(.*?)</args>', response, re.DOTALL)

    if not tool_match:
        return None, None

    tool_name = tool_match.group(1).strip()
    args = {}
    if args_match:
        try:
            args = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError:
            return tool_name, {}

    return tool_name, args


def run_agent(model, tokenizer, conversation_history, streamer, max_turns=5):
    """
    Agent loop — generates a response, checks for tool calls, executes them and feeds results back until no more tool calls
    max_turns prevents infinite loops
    """
    # combine base system prompt with tool descriptions
    full_prompt = SYSTEM_PROMPT + TOOL_DESCRIPTIONS

    for turn in range(max_turns):
        response = generate_response(
            model, tokenizer, conversation_history, full_prompt, streamer
        )

        tool_name, args = parse_tool_call(response)

        if not tool_name:
            # no tool call — this is the final response
            return response

        # execute the tool
        if tool_name == "tool_help":
            tool_result = tool_help(**args)
        elif tool_name not in TOOLS:
            tool_result = f"[ERROR] Unknown tool: {tool_name}"
        else:
            try:
                tool_result = TOOLS[tool_name](**args)
            except Exception as e:
                tool_result = f"[ERROR] Tool execution failed: {e}"

        if DEBUG:
            print(f"\n[TOOL] {tool_name}({args})")
            print(f"[RESULT] {tool_result[:200]}")

        # feed result back so model can continue reasoning
        conversation_history.append({"role": "assistant", "content": response})
        conversation_history.append({
            "role": "user",
            "content": f"<tool_result>{tool_result}</tool_result>"
        })

    return "[ERROR] Max agent turns reached without final response"