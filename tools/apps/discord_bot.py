'''
Discord bot integration
- Runs as a background thread alongside the tray app
- Shares the same model instance for consistent responses
- Knowledge base access only — no file system or tool execution for security
- Logs all conversations for fine-tuning data collection
'''

from pathlib import Path
import asyncio
import discord
import threading
import os
from datetime import datetime
from dotenv import load_dotenv
from config import DEBUG, LOGS_DIR, DISCORD_SYSTEM_PROMPT
import json
from tools.knowledge import search_knowledge_base
from core.model import generate_response

load_dotenv()  # load DISCORD_TOKEN from .env file

# shared model reference — set by the tray on startup
_model = None
_tokenizer = None
_rag = None

# per-user conversation history for Discord key: user_id, value: list of message dicts
discord_histories = {}

MAX_DISCORD_HISTORY = 20  # keep last N messages per user


def set_discord_model(model, tokenizer, rag):
    """
    Register the model, tokenizer and RAG instance for Discord use.
    Called once on startup by TrayApp.initialise()

    model: the loaded language model
    tokenizer: the model's tokenizer
    rag: the RAG instance for knowledge base access
    """
    global _model, _tokenizer, _rag
    _model = model
    _tokenizer = tokenizer
    _rag = rag


def save_discord_session(user_id, username, conversation):
    """
    Overwrite the current session file for this user after every message
    Prevents log loss if bot crashes or is force-stopped

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
    Save a final final timestamped log of the conversation for fine-tuning data collection

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
        current.unlink()

    if DEBUG:
        print(f"[DISCORD] Conversation saved to {path}")


def generate_discord_response(user_id, username, user_message):
    """
    Generate a response for a Discord message without any external tool access for security

    user_id: Discord user ID for conversation tracking
    user_message: the message string from the Discord user
    returns: response string
    """
    if _model is None or _tokenizer is None:
        return "I'm still loading, please try again in a moment!"

    # get or create conversation history for this user
    if user_id not in discord_histories:
        discord_histories[user_id] = []

    history = discord_histories[user_id]
    history.append({"role": "user", "content": user_message})

    # simple heuristic — if message looks like a question about something specific, search knowledge base
    should_search = _rag is not None and (
        "?" in user_message or
        any(kw in user_message.lower() for kw in ["what", "how", "tell me", "explain", "find"])
    )

    if should_search:
        log_result = search_logs_for_user(user_message, username)
        if not log_result.startswith("[INFO]") and not log_result.startswith("[ERROR]"):
            augmented_history = history[:-1] + [{
                "role": "user",
                "content": f"{log_result}\n\nUser question: {user_message}"
            }]
        else:
            kb_result = search_knowledge_base(user_message)
            if not kb_result.startswith("[INFO]") and not kb_result.startswith("[ERROR]"):
                augmented_history = history[:-1] + [{
                    "role": "user",
                    "content": f"{kb_result}\n\nUser question: {user_message}"
                }]
            else:
                augmented_history = history
    else:
        augmented_history = history

    # use shared generate_response with Discord system prompt
    response = generate_response(
        _model, _tokenizer,
        augmented_history,
        DISCORD_SYSTEM_PROMPT,
        streamer=None # no streaming for Discord, just return full response
    )

    history.append({"role": "assistant", "content": response})

    if len(history) > MAX_DISCORD_HISTORY * 2:
        discord_histories[user_id] = history[-(MAX_DISCORD_HISTORY * 2):]

    return response

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


class MarvinBot(discord.Client):
    """
    Discord bot client for Marvin
    Responds to all messages in the server
    """

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # required to read message content
        super().__init__(intents=intents)

    async def on_ready(self):
        print(f"[DISCORD] Marvin is online as {self.user}")

    async def on_message(self, message):
        # ignore messages from the bot itself
        if message.author == self.user:
            return

        # ignore messages from other bots
        if message.author.bot:
            return

        if DEBUG:
            print(f"[DISCORD] {message.author.name}: {message.content}")

        # handle special commands
        content = message.content.strip().lower()

        if content == "!clear":
            history = discord_histories.pop(message.author.id, [])
            if history:
                # save timestamped log before clearing
                save_discord_log(message.author.id, message.author.name, history)
            await message.channel.send("Your conversation history with me has been cleared!")
            return

        if content == "!help":
            await message.channel.send(
                "**Marvin Commands:**\n"
                "`!clear` - clear your conversation history with me\n"
                "`!help` - show this message\n\n"
                "I can answer questions, search documents and help with general topics.\n"
                "What I can't do yet: use search engine, see chat history with other users \n"
                "Just chat normally - no commands needed!"
            )
            return

        # run model inference in a thread so the asyncio event loop stays responsive
        async with message.channel.typing():
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                generate_discord_response,
                message.author.id,
                message.author.name,
                message.content
            )

        if DEBUG:
            print(f"[DISCORD] Response ({len(response)} chars): {response[:100]}")

        history = discord_histories.get(message.author.id, []) # save history after each message
        save_discord_session(message.author.id, message.author.name, history)

        # Discord has a 2000 character limit
        if len(response) > 1900:
            # split into chunks
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(response)

    async def get_channel_summary(self, channel_id, limit=100):
        """
        Get a summary of recent messages in a channel for context

        channel_id: ID of the Discord channel to summarize
        limit: how many recent messages to include in the summary
        returns: summary string
        """
        channel = self.get_channel(channel_id)
        if not channel:
            return "[ERROR] Channel not found"

        messages = await channel.history(limit=limit).flatten()
        content = "\n".join(f"{msg.author.name}: {msg.content}" for msg in messages[::-1])
        summary_prompt = f"Summarize the following conversation from Discord:\n\n{content}\n\nSummary:"
        
        # use shared generate_response with Discord system prompt
        summary = generate_response(
            _model, _tokenizer,
            [{"role": "user", "content": summary_prompt}],
            DISCORD_SYSTEM_PROMPT,
            streamer=None
        )
        return summary

def start_discord_bot():
    """
    Start the Discord bot in a background thread.
    Called from TrayApp.initialise() after model loads.
    """
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("[DISCORD] No DISCORD_TOKEN found in .env — Discord bot disabled")
        return

    def run_bot():
        import asyncio
        bot = MarvinBot()
        asyncio.run(bot.start(token))

    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()
    print("[DISCORD] Bot starting...")