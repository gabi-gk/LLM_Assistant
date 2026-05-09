'''
Lets the model call tools by including tool tags in the generated response
- the respose is parsed for tool calls and arguments
- the tool is executed and the result is fed back to the model in a tool_result tag
- the model can then continue generating a response based on the tool result, allowing it to use multiple commands in a single request
'''

import json
import re
from config import DEBUG, SYSTEM_PROMPT
from agent.registry import TOOLS, TOOL_DESCRIPTIONS, tool_help
from core.model import generate_response

def parse_tool_call(response):
    """
    Check if the model's response contains a tool call

    response: the raw response string from the model, which may contain tool call tags
    returns (tool_name, args_dict) or none 
    """
    tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL) # check for <tool> tags in the response and extract the tool name
    args_match = re.search(r'<args>(.*?)</args>', response, re.DOTALL) # check for <args> tags and extract the JSON string of arguments

    if not tool_match or not args_match:
        return None, None

    tool_name = tool_match.group(1).strip() # extract the tool name from the tags 
    args = {}
    if args_match:
        try:
            raw = args_match.group(1).strip() # extract the raw JSON string of arguments from the tags
            # find the actual boundaries and remove any extra text to avoid JSON parsing errors
            start = raw.find('{') 
            end = raw.rfind('}') + 1
            if start != -1 and end > start:
                args = json.loads(raw[start:end]) # extract the JSON substring containing the arguments
        except json.JSONDecodeError:
            return tool_name, {}

    if DEBUG:
        print(f"[DEBUG] args type: {type(args)}, value: {args}")
    return tool_name, args


def run_agent(model, tokenizer, conversation_history, streamer, max_turns=5):
    """
    Generates a response, checks for tool calls, executes them and feeds results back until no more tool calls
    
    model: the loaded language model
    tokenizer: the model's tokenizer
    conversation_history: list of dicts with 'role' and 'content' keys representing the conversation history
    streamer: the model's streamer for token-by-token generation
    max_turns: maximum number of tool calls before giving up and returning an error
    """
    # combine base system prompt with tool descriptions
    full_prompt = SYSTEM_PROMPT + TOOL_DESCRIPTIONS
    
    # run each time the assistant is prompted for a response
    for turn in range(max_turns):
        response = generate_response( # gemerate a response from the model given the conversation history and the full prompt 
            model, tokenizer, conversation_history, full_prompt, streamer
        )

        tool_name, args = parse_tool_call(response) # extract any tool call and arguments from the response

        if not tool_name:
            # If there is no tool call, return the response to be displayed in the chat window
            return response

        # execute the tool
        if tool_name not in TOOLS:
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