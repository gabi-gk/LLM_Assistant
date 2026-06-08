'''
Lets the model call tools by including tool tags in the generated response
- the respose is parsed for tool calls and arguments
- the tool is executed and the result is fed back to the model in a tool_result tag
- the model can then continue generating a response based on the tool result, allowing it to use multiple commands in a single request
'''

import json
import re
from config import DEBUG, SYSTEM_PROMPT
from agent.registry import TOOLS, TOOL_DESCRIPTIONS
from core.model import generate_response
from core.utils import get_system_prompt

# how many consecutive failed tool calls before the loop stops the model brute forcing
MAX_CONSECUTIVE_ERRORS = 3
# how many tool_help calls allowed per request before forcing a direct answer
MAX_TOOL_HELP_CALLS = 2


def parse_tool_call(response):
    """
    Check if the model's response contains a tool call

    response: the raw response string from the model, which may contain tool call tags
    returns (tool_name, args_dict) or none 
    """
    tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL) # check for <tool> tags in the response and extract the tool name
    args_match = re.search(r'<args>(.*?)</args>', response, re.DOTALL) # check for <args> tags and extract the JSON string of arguments

    if tool_match and not args_match:
        return tool_match.group(1).strip(), "__MISSING_ARGS__" # flag missing args

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
        except json.JSONDecodeError: # return a sentiel indication rather than epty args
            return tool_name, "__JSON_ERROR__"

    if DEBUG:
        print(f"[DEBUG] args type: {type(args)}, value: {args}")
    return tool_name, args

def extract_tool_block(response):
    """
    Pull out just the tool invocation (tool + args tags) from a response

    response: the raw model response containing a tool call
    returns: the clean tool call string
    """
    tool_match = re.search(r'<tool>.*?</tool>', response, re.DOTALL)
    args_match = re.search(r'<args>.*?</args>', response, re.DOTALL)
    if tool_match and args_match:
        return f"{tool_match.group(0)}\n{args_match.group(0)}"
    return response

def final_answer(model, tokenizer, working, full_prompt, streamer, system_msg):
    """
    Force the model to give a final answer without more tool calls by appending a system instruction

    system_msg: the instruction describing why it should stop
    returns: the model's final reply string
    """
    working.append({"role": "user", "content": f"<system>{system_msg}</system>"})
    return generate_response(model, tokenizer, working, full_prompt, streamer)

def record(working, response, tool_result):
    """
    Store only the clean result in model history 

    working: the current model history to append to
    response: the raw model response containing the tool call
    tool_result: the result of executing the tool, to be fed back to the model
    """
    working.append({"role": "assistant", "content": extract_tool_block(response)})
    working.append({"role": "user", "content": f"<tool_result>{tool_result}</tool_result>"})

def run_agent(model, tokenizer, conversation_history, streamer, max_turns=8):
    """
    Generates a response, checks for tool calls, executes them and feeds results back until no more tool calls
    
    model: the loaded language model
    tokenizer: the model's tokenizer
    conversation_history: list of dicts with 'role' and 'content' keys representing the conversation history
    streamer: the model's streamer for token-by-token generation
    max_turns: maximum number of tool calls before giving up and returning an error
    """
    # combine base system prompt with tool descriptions and add current time
    full_prompt = get_system_prompt(SYSTEM_PROMPT) + TOOL_DESCRIPTIONS
    
    # make a working copy of the conversation history to append tool results to without modifying the original
    working = list(conversation_history)
    
    consecutive_errors = 0 # track the failed tool calls to stop the brute forcing
    tool_help_calls = 0 # prevent infinite tool help calls

    # run each time the assistant is prompted for a response
    for turn in range(max_turns):
        response = generate_response( # gemerate a response from the model given the model history and the full prompt 
            model, tokenizer, working, full_prompt, streamer
        )

        tool_name, args = parse_tool_call(response) # extract any tool call and arguments from the response

        if not tool_name:
            # If there is no tool call, return the response to be displayed in the chat window
            return response

        # Handle the JSON parsing errors
        if args == "__JSON_ERROR__":
            consecutive_errors += 1
            record(working, response, f"[ERROR] Invalid JSON in args for {tool_name}. Check your quoting and escaping.")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return final_answer(model, tokenizer, working, full_prompt, streamer, "The last tool calls failed. Do not attempt more tool calls. Tell the user plainly what failed, then stop.")
            continue
        elif args == "__MISSING_ARGS__":
            consecutive_errors += 1
            record(working, response, f"[ERROR] Tool call for {tool_name} is missing its <args> block. Provide both <tool> and <args>.")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return final_answer(model, tokenizer, working, full_prompt, streamer, "The last tool calls were malformed. Stop and tell the user plainly what you couldn't do.")
            continue

        # execute the tool
        if tool_name not in TOOLS:
            tool_result = f"[ERROR] Unknown tool: {tool_name}"
        else:
            try:
                tool_result = TOOLS[tool_name](**args)
            except Exception as e:
                tool_result = f"[ERROR] Tool execution failed: {e}"

        tool_result = str(tool_result) # string for the error check 

        if tool_result.startswith("[BLOCKED]"):
            record(working, response, tool_result)
            return final_answer(model, tokenizer, working, full_prompt, streamer, "This action is blocked for safety and cannot be run. Tell the user plainly that it's destructive and you can't execute it, and why.")

        error = tool_result.startswith("[ERROR]") or tool_result.startswith("[BLOCKED]") # check if the tool execution resulted in an error
        
        if tool_name == "tool_help":
            tool_help_calls += 1  # keep track of tool help 
            if tool_help_calls > MAX_TOOL_HELP_CALLS:
                record(working, response, tool_result)
                return final_answer(model, tokenizer, working, full_prompt, streamer, "Stop calling tool_help. Use what you know to answer the user directly, or tell the user you can't.")
        elif tool_result.startswith("[ERROR]"):
            consecutive_errors += 1 # increment on a real failure
        else:
            consecutive_errors = 0 # reset on success
        
        if DEBUG:
            print(f"\n[TOOL] {tool_name}({args})")
            print(f"[RESULT] {tool_result[:200]}")

        # feed the clear result back so model can continue reasoning
        record(working, response, tool_result)

        # A simple block to prevent the model from brute forcing instead of admitting failure.
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return final_answer(model, tokenizer, working, full_prompt, streamer, "The last tool calls failed. Do not attempt more tool calls. Tell the user plainly what failed, then stop.")

    return "[ERROR] I got stuck trying to reach a response."