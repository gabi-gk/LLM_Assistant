'''
Discord bot integration
- Runs as a background thread alongside the tray app
- Shares the same model instance for consistent responses
- Knowledge base access only — no file system or tool execution for security
- Logs all conversations for fine-tuning data collection
- Full chat history access
- Mentions and pings via a custom tool that looks up guild members by name
- Admin command to toggle between responding to all messages vs mention only
'''

from pathlib import Path
import asyncio
import discord
import threading
import os
import json
import re
from dotenv import load_dotenv
from tools.knowledge import search_knowledge_base
from core.model import generate_response
from agent.registry import get_discord_tools, DISCORD_TOOL_DESCRIPTIONS
from config import DEBUG, LOGS_DIR, DISCORD_SYSTEM_PROMPT
from agent.loop import parse_tool_call
from datetime import datetime, timezone, timedelta

load_dotenv()  # load DISCORD_TOKEN from .env file

# shared model reference - set by the tray on startup
_model = None
_tokenizer = None
_rag = None
mention_only = True # used to switch between respond to all messages vs mention only

owner_ids = set() # get the admins who can use special commands from the env
raw_ids = os.getenv("DISCORD_OWNER_IDS", "")
if raw_ids:
    owner_ids = {int(id.strip()) for id in raw_ids.split(",") if id.strip()}

# per-user conversation history for Discord key: user_id, value: list of message dicts
discord_histories = {}

current_channel = None  # track the current channel for deep_history
bot_loop = None # bot's event loop for running async tasks from sync code

MAX_DISCORD_HISTORY = 20  # keep last N messages per user


def set_discord_model(model, tokenizer, rag):
    """
    Register the model, tokenizer and RAG instance for Discord use 

    model: the loaded language model
    tokenizer: the model's tokenizer
    rag: the RAG instance for knowledge base access
    """
    global _model, _tokenizer, _rag
    _model = model
    _tokenizer = tokenizer
    _rag = rag

def start_discord_bot():
    """
    Start the Discord bot in a background thread
    """
    token = os.getenv("DISCORD_TOKEN") # get the bot token from env
    if not token:
        print("[DISCORD] No DISCORD_TOKEN found in .env — Discord bot disabled")
        return

    def run_bot():
        global bot_loop
        bot_loop = asyncio.new_event_loop() # create a new event loop for the bot thread
        asyncio.set_event_loop(bot_loop)
        bot = MarvinBot()
        bot_loop.run_until_complete(bot.start(token))

    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()
    print("[DISCORD] Bot starting...")

def save_discord_session(user_id, username, conversation):
    """
    Overwrite the current session file for this user after every message to prevent log loss

    user_id: Discord user ID
    username: Discord username
    conversation: current conversation history list
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    # one file per user, overwritten each message
    path = f"{LOGS_DIR}/discord_current_{username}.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "platform": "discord",
            "user_id": str(user_id),
            "username": username,
            "conversation": conversation
        }, f, indent=2)

def save_discord_log(user_id, username, conversation):
    """
    Save a final final timestamped log of the conversation for fine-tuning data collection, after clearing history 

    user_id: Discord user ID
    username: Discord username
    conversation: conversation history to save
    """
    if not conversation:
        return

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{LOGS_DIR}/discord_{username}_{timestamp}.json"

    with open(path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "platform": "discord",
            "user_id": str(user_id),
            "username": username,
            "conversation": conversation
        }, f, indent=2)

    current = Path(f"{LOGS_DIR}/discord_current_{username}.json")
    if current.exists():
        current.unlink() # remove the current session file after saving the final log to prevent mix ups

    if DEBUG:
        print(f"[DISCORD] Conversation saved to {path}")

def deep_history(limit=100, username_filter=None, hours=None):
    """
    Fetch and summarise deeper channel history.
    Optionally filter to a specific username.

    limit: how many messages to fetch
    username_filter: if provided, only include messages from this user
    hours: if provided, only include messages from the last N hours
    returns: summarised string or specific user messages
    """
    if current_channel is None or bot_loop is None:
        return "[INFO] No channel available"

    future = asyncio.run_coroutine_threadsafe( # run the async message fetching in the bot's event loop
        message_by_time(current_channel, limit, hours), bot_loop
    )
    try:
        messages = future.result(timeout=10) # wait for the messages to be fetched, with a timeout to prevent hanging
    except Exception as e:
        return f"[ERROR] Failed to fetch channel history: {e}"

    if username_filter: 
        # filter to specific user
        user_messages = [
            msg for msg in messages
            if username_filter.lower() in msg.author.display_name.lower() # filter by display name or username to be flexible with mentions
            or username_filter.lower() in msg.author.name.lower()
        ]
        if not user_messages:
            return f"[INFO] No messages found from '{username_filter}' in the last {limit} messages"
        
        lines = [f"{msg.author.display_name}: {msg.content}" 
                 for msg in user_messages if msg.content] # get all messages with content and format with author's display name
        return f"[Messages from {username_filter}:]\n" + "\n".join(lines)

    # full history summary
    content = "\n".join(
        f"{'Marvin' if msg.author.bot else msg.author.display_name}: {msg.content[:150]}"
        for msg in messages if msg.content
    )

    if not content:
        return "[INFO] No messages found"

    summary = generate_response( # model summarizes the history for getting context without data overload
        _model, _tokenizer,
        [{"role": "user", "content": f"Summarise this Discord conversation concisely:\n\n{content}"}],
        DISCORD_SYSTEM_PROMPT,
        streamer=None
    )
    return summary

async def message_by_time(channel, limit, hours = None):
    '''
    Fetch messages from a channel, optionally filtering to a specific time range

    channel: the Discord channel to fetch from
    limit: how many messages to fetch
    hours: if provided, only include messages from the last N hours
    returns: list of messages sorted from oldest to newest
    '''
    after = None
    if hours:
        after = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    messages = [msg async for msg in channel.history(limit=limit, after=after)] # fetch messages with optional time filter
    messages.reverse() # reverse to get oldest to newest
    return messages

def generate_discord_response(user_id, username, user_message, channel_context=""):
    '''
    Respond to a Discord message - modfied from the main generate_response to include channel context and use the Discord tool registry
    
    user_id: Discord user ID for tracking conversation history
    username: Discord username for logging and personalized responses
    user_message: the content of the user's message to respond to
    channel_context: optional string of recent channel messages for additional context
    returns: the generated response string to send back to Discord
    '''
    if _model is None or _tokenizer is None:
        return "I'm still loading, please try again in a moment!"

    # track the conversation history for the user, each has their own history
    if user_id not in discord_histories:
        discord_histories[user_id] = []

    history = discord_histories[user_id]

    # build message with channel context
    if channel_context:
        full_message = (
            f"[Recent channel messages:]\n{channel_context}\n\n"
            f"Current message from {username}: {user_message}"
        )
    else:
        full_message = user_message

    # working history for this turn - don't mutate discord_histories yet
    working_history = history + [{"role": "user", "content": full_message}]

    discord_tools = get_discord_tools(
        search_kb_func=search_knowledge_base,
        deep_history_func=deep_history,
        mention_user_func=mention_user
    )

    full_prompt = DISCORD_SYSTEM_PROMPT + DISCORD_TOOL_DESCRIPTIONS
    final_response = None

    # try to generate a response with up to 3 calls
    for turn in range(3):
        response = generate_response(
            _model, _tokenizer,
            working_history,
            full_prompt,
            streamer=None
        )

        tool_name, args = parse_tool_call(response) # check if the model is asking to use a tool

        if not tool_name:
            final_response = response
            break

        if tool_name not in discord_tools:
            tool_result = f"[ERROR] Tool not available on Discord: {tool_name}"
        else:
            try:
                tool_result = discord_tools[tool_name](**args)
            except Exception as e:
                tool_result = f"[ERROR] Tool execution failed: {e}"

        if DEBUG:
            print(f"[DISCORD TOOL] {tool_name}({args})")
            print(f"[DISCORD RESULT] {tool_result[:200]}")

        working_history.append({"role": "assistant", "content": response})
        working_history.append({
            "role": "user",
            "content": f"<tool_result>{tool_result}</tool_result>"
        })

    if final_response is None:
        final_response = "[I reached my turn limit — please try rephrasing your question]"

    mention_pattern = re.compile(r'<@\d+>')
    for msg in working_history:
        if isinstance(msg.get("content"), str) and "<tool_result>" in msg["content"]:
            mentions = mention_pattern.findall(msg["content"])
            for mention in mentions:
                if mention not in final_response:
                    final_response = f"{mention} {final_response}"


    # save clean exchange to persistent history
    # store user_message not full_message so logs are readable
    discord_histories[user_id] = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_response}
    ]

    if len(discord_histories[user_id]) > MAX_DISCORD_HISTORY * 2:
        discord_histories[user_id] = discord_histories[user_id][-(MAX_DISCORD_HISTORY * 2):]

    return final_response

def mention_user(username):
    """
    Find a guild member by name and return their Discord mention string.
    The bot can use this to ping someone in a message.
    
    username: partial or full display name or username to search for
    returns: Discord mention string e.g. <@123456789> or error message
    """
    if current_channel is None or bot_loop is None:
        return "[INFO] No channel available"
    
    future = asyncio.run_coroutine_threadsafe(
        find_member(current_channel.guild, username), bot_loop
    )
    try:
        member = future.result(timeout=5)
    except Exception as e:
        return f"[ERROR] Could not find member: {e}"
    
    if member is None:
        return f"[INFO] No member found matching '{username}'"
    
    return f"<@{member.id}>"  # Discord mention format


async def find_member(guild, username):
    """
    Search guild members for a name match to find the right person to ping - exact or partial match
    
    guild: Discord guild object
    username: name to search for
    returns: Member object or None
    """
    if guild is None:
        return None
    
    name_lower = username.lower()
    
    # exact match first
    for member in guild.members:
        if (member.display_name.lower() == name_lower or 
            member.name.lower() == name_lower):
            return member
    
    # partial match
    for member in guild.members:
        if (name_lower in member.display_name.lower() or 
            name_lower in member.name.lower()):
            return member
    
    return None

async def recent_channel_context(channel, limit=20):
    '''
    Get the last N messages for context for the AI

    channel: the Discord channel to fetch from
    limit: how many messages to fetch
    returns: formatted string of recent messages to be included in the model input 
    '''
    messages = [msg async for msg in channel.history(limit=limit)]
    messages.reverse()
    
    lines = []
    for msg in messages:
        if not msg.content:
            continue
        name = "You (Marvin)" if msg.author.bot else msg.author.display_name
        # truncate long bot messages so summaries don't dominate context
        content = msg.content[:150] + "..." if len(msg.content) > 150 and msg.author.bot else msg.content[:150]
        lines.append(f"{name}: {content}")
    
    return "\n".join(lines)

class MarvinBot(discord.Client):
    """
    Discord bot client for Marvin AI that responds to all messages in the guild
    """

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # required to read message content
        intents.members = True  # required to fetch member info for mentions
        super().__init__(intents=intents)

    async def on_ready(self):
        '''
        Called when the bot is ready and connected to Discord
        '''
        print(f"[DISCORD] Marvin is online as {self.user}")

    async def on_message(self, message):
        '''
        Waits for messages and responds when appropriate, with the model's response to the message content and recent channel context

        message: the Discord message object that triggered this event
        '''
        global current_channel
        global mention_only
        current_channel = message.channel

        # ignore messages from the bot itself or other bots to prevent loops
        if message.author == self.user:
            return
        if message.author.bot:
            return

        if DEBUG:
            print(f"[DISCORD] {message.author.name}: {message.content}")

        content = message.content.strip().lower()

        if content == "!clear": # definition of the clear command to clear conversation history for a user
            history = discord_histories.pop(message.author.id, [])
            if history:
                save_discord_log(message.author.id, message.author.name, history)
            await message.channel.send("Your conversation history with me has been cleared!")
            return

        if content == "!help": # definition of the help command to list available commands and usage info
            await message.channel.send(
                "**Marvin Commands:**\n"
                "`!clear` - clear your conversation history with me\n"
                "`!help` - show this message\n"
                "`!toggle` - (Admin only) switch between responding to everything vs mentions only\n\n"
                "I can answer questions, search documents and help with general topics.\n"
                "Just chat normally - no commands needed!"
            )
            return

        if content == "!toggle": # definition of the toggle command to switch between response modes
            if owner_ids and message.author.id not in owner_ids: # only admins can toggle modes
                await message.channel.send("Only guild owners can toggle my response mode.")
                return
            mention_only = not mention_only
            mode = "mention only (@Marvin or 'marvin')" if mention_only else "responding to everything"
            await message.channel.send(f"Switched to **{mode}** mode.")
            return

        # check if the model should respond
        if mention_only:
            mentioned = (
                self.user in message.mentions or # @Marvin ping
                "marvin" in content # name mentioned
            )
            if not mentioned:
                return 

        # fetch recent channel context before generating
        channel_context = await recent_channel_context(message.channel)

        async with message.channel.typing():
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                generate_discord_response,
                message.author.id,
                message.author.name,
                message.content,
                channel_context  # pass channel context
            )

        if DEBUG:
            print(f"[DISCORD] Response ({len(response)} chars): {response[:100]}")

        history = discord_histories.get(message.author.id, [])
        save_discord_session(message.author.id, message.author.name, history) # save the sessions

        if len(response) > 1900: # truncate long responses to fit Discord limits, splitting into multiple messages if needed
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(response)


# Currently not used - to be added after finetuning the model with tool calls to avoid confusion
def search_logs_for_user(query, username):
    """
    Search conversation logs filtered to a specific Discord user for relevant information to include in the model input context

    query: search query string
    username: Discord username to filter logs by
    returns: formatted context string or [INFO] if nothing found
    """
    if _rag is None:
        return "[INFO] No knowledge base available"

    all_chunks = _rag.collection.get(include=["documents", "metadatas"])

    matches = []
    logs_path = Path(LOGS_DIR).resolve()
    query_lower = query.lower()
    words = [w for w in query_lower.split() if len(w) > 2]

    for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]): # iterate through all chunks in the RAG collection
        source = meta.get("source", "")
        source_path = Path(source).resolve()

        # only match logs from this specific user
        if not (logs_path in source_path.parents or source_path.parent == logs_path):
            continue
        if "discord_" not in source_path.name:
            continue
        if username.lower() not in source_path.name.lower():
            continue

        # keyword match
        doc_lower = doc.lower()
        hits = sum(1 for w in words if w in doc_lower)
        if hits >= max(1, len(words) // 2):
            matches.append({
                "text": doc,
                "source": source,
                "type": meta.get("type", "text"),
                "name": source_path.name,
                "similarity": round(hits / max(len(words), 1), 2)
            })

    if not matches:
        return "[INFO] No relevant information found in your conversation logs"

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return _rag.format_context(matches[:5])