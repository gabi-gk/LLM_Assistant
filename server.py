'''
 Open-AI compatible server for Marvin

- loads the model once at startup and keeps it in memory for every request
- uses /v1/chat/completions work with OpenAIs chat format
- uses GET /health which reports whether the model is loading, ready or failed
- requests go straight to the model - no agent loop, no tools, no self model, no RAG so it only generates swappable text completions
- paths starting /v1/ are OpenAI's 
- Streaming is not supported yet - requests with "stream": true are rejected
'''

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # same as run.py/main.py

import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SERVER_HOST, SERVER_PORT, TRAINED_MODEL, BASE_MODEL, ensure_dirs
from core.model import load_model, generate_response
from core.lock import acquire_lock, release_lock

# model name - no problems for one model
MODEL_ID = "marvin"

class ServerState:
    '''
    Setup the server state that is used by all other threads 
    '''
    def __init__(self):
        self.status = "starting" # starting -> loading -> ready, or -> error
        self.error = None # the error message if loading failed
        self.model = None
        self.tokenizer = None
        self.loaded_from = None # which path the weights actually came from (trained or base fallback)
        self.started_at = time.time()
        # Make sure there is no more then one request at a time since thwo requests would corrupt each other's key value cache and exceed VRAM
        self.generate_lock = threading.Lock()

state = ServerState() # create the state shared by all instances


def load_everything():
    '''
    Load the model and tokenizer - the slow part of startup
    Runs in a background thread so the server can already answer /health with "loading" while this works
    '''
    try:
        state.status = "loading"

        print("[SERVER] Loading model...")
        state.model, state.tokenizer = load_model()
        # load_model() silently falls back to the base model if the trained one is missing, mirror that check here so /health can report it
        state.loaded_from = TRAINED_MODEL if os.path.exists(TRAINED_MODEL) else BASE_MODEL

        state.status = "ready" # only flip to ready once loading has succeeded
        print(f"[SERVER] Ready - model loaded from {state.loaded_from}")
    except Exception as e:
        # keep the server running so /health can explain what went wrong instead of the process just dying
        state.status = "error"
        state.error = f"{type(e).__name__}: {e}"
        print(f"[SERVER] Startup failed: {state.error}")


@asynccontextmanager 
async def lifespan(app):
    '''
    called by uvicorn when the server starts, before it starts accepting requests
    load the model in a background thread and allow it to use /health to report status
    '''
    threading.Thread(target=load_everything, daemon=True).start() # create a background thread to load the model, daemon so a half-loaded model doesn't block shutting down the server
    yield # break point between startup and shutdown
    print("[SERVER] Shutting down") 


app = FastAPI(title="Marvin", lifespan=lifespan) # actually create the app


class ChatMessage(BaseModel):
    '''
    Validates an incoming message in the OpenAI chat format
    '''
    role: Literal["system", "user", "assistant", "tool"] # restrict the roles so the fields can always be validated
    content: Optional[str] = None  # the text of the message can be absent or null

class ChatCompletionRequest(BaseModel):
    '''
    Rules for FastAPI's request parsing to match the OpenAI chat format
    Anything incorrectly shaped returns a 422 error
    '''
    model: str = MODEL_ID # only one model so client input is not used
    messages: list[ChatMessage] # the conversation history, must end with a user message
    stream: bool = False # not supported

# NOTE: FastAPI and handler errors are different, 422 comes from FastAPI automatically

def openai_error(status_code, message, err_type="invalid_request_error"):
    '''
    Called when the server wants to match OpenAI's error format
    '''
    return JSONResponse( # return a JSON response with the correct format
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )



@app.get("/health")
def health():
    '''
    Report whether the server is ready to take chat requests
    Returns 200 only when ready, 503 otherwise 
    '''
    body = {
        "status": state.status,
        "model_loaded": state.model is not None,
        "loaded_from": state.loaded_from,
        "busy": state.generate_lock.locked(), # True while a request is generating - request will queue
        "uptime_seconds": round(time.time() - state.started_at),
    }
    if state.error: # only included if the server failed to load
        body["error"] = state.error
    return JSONResponse(status_code=200 if state.status == "ready" else 503, content=body)


@app.get("/v1/models")
def list_models():
    '''
    Many OpenAI clients call this first to find out which models exist
    Right now only one model is supported
    '''
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(state.started_at), "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    '''
    The main endpoint - takes an OpenAI chat request, runs a single generation on the model
    and wraps the reply in an OpenAI chat.completion object
    '''
    # refuse early rather than queueing requests that can't be served
    if state.status != "ready":
        return openai_error(503, f"Model is not ready (status: {state.status})", "server_error")
    if body.stream:
        return openai_error(400, "Streaming is not supported yet, send \"stream\": false")

    # Qwen's chat template doesn't allow multiple system messages so join the system messages into one string
    system_prompt = "\n\n".join(m.content for m in body.messages if m.role == "system" and m.content) # this is empty if the user didn't send any system messages

    # Create a plain history of all user and Marvin messages, not including system messages or tool calls
    history = [
        {"role": m.role, "content": m.content or ""}
        for m in body.messages
        if m.role in ("user", "assistant")
    ]
    if not history or history[-1]["role"] != "user": # Model can't generate a reply to his own message so the last message must be from the user
        return openai_error(400, "messages must end with a user message")

    # if user forces close, the genarte stops early and returns none
    cancel = threading.Event()

    def generate():
        '''
        calls the model to generate a response but only if the GPU is free, else it queues
        '''
        with state.generate_lock: # wait for the GPU
            if cancel.is_set(): # client disconnected abandon generation
                return None
            # streaing is not supported yet
            return generate_response(state.model, state.tokenizer, history, system_prompt, streamer=None, cancel=cancel)

    async def watch_disconnect():
        '''
        If the client disconnects stop the generation and return a 499 error
        '''
        while True:
            if await request.is_disconnected():
                cancel.set()
                return
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(watch_disconnect()) # watch in the background to check if the client disconnects during the generation
    try:
        reply = await asyncio.to_thread(generate) # run the generation in a thread so the async server can still answer /health while the GPU is busy
    except Exception as e: # catch any error from the model and return it in OpenAI's format instead of crashing the server
        return openai_error(500, f"Generation failed: {type(e).__name__}: {e}", "server_error")
    finally:
        watcher.cancel() # done - stop the watcher

    if reply is None or cancel.is_set():
        # if the client disconnected, don't generate a reply
        return openai_error(499, "Request cancelled by client")

    # count the token so users can track their usage
    prompt_text = state.tokenizer.apply_chat_template( # use the template to get the messages that were sent to the model
        [{"role": "system", "content": system_prompt}] + history,
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_tokens = len(state.tokenizer.encode(prompt_text)) # tokens in the prompts
    completion_tokens = len(state.tokenizer.encode(reply)) # tokens in the reply

    # return in the OpenAI format so any OpenAI client library can use it
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    ensure_dirs() # create any missing directories at runtime
    acquire_lock() # ensure only one instance runs at a same time
    try:
        # No authentication, so only localhost works
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT) # start the event loop
    finally:
        release_lock()
