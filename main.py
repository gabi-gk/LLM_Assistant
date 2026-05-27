'''
Terminal debug interface minimal version
Does not include self model, full history and confirmations

Use run.py for general use
'''

from config import COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT
from core.model import load_model, create_streamer
from core.history import compact_history, save_conversation, load_last_session, save_session_state
from core.rag import RAG
from agent.loop import run_agent
from tools.notifications import restore_reminders
from core.lock import acquire_lock, release_lock
from tools.knowledge import set_rag
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_chat():
    '''
    The main chat loop for the terminal interface
    single-thread version of the app.py script, without the GUI elements
    '''

    conversation_history = load_last_session()
    
    print("Loading model...")
    model, tokenizer = load_model()
    streamer = create_streamer(tokenizer)

    rag = RAG()
    rag.index_all() # index any new data for RAG
    set_rag(rag)

    restore_reminders() # check for any acive reminders and restore them

    print("Ready. Type 'quit' to exit, 'clear' to reset conversation history.\n")
    
    # Let the user type
    while True:
        # compact the history 
        conversation_history = compact_history(
            model, tokenizer, conversation_history,
            threshold=COMPACTION_THRESHOLD,
            keep_recent=COMPACTION_KEEP_RECENT
        )

        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            save_conversation(conversation_history)
            save_session_state("closed")
            break
        # handle pre-defined special commands
        if user_input.lower() == "clear": # clear conversation history and session state
            save_conversation(conversation_history)
            save_session_state("cleared")
            conversation_history = []
            print("History cleared.\n")
            continue
        
        # save history
        conversation_history.append({"role": "user", "content": user_input})
    
        # AI response
        print("AI: ", end="", flush=True)
        reply = run_agent(model, tokenizer, conversation_history, streamer)
        
        conversation_history.append({"role": "assistant", "content": reply})
        print()

if __name__ == "__main__":
    acquire_lock()
    try:
        run_chat()
    finally:
        release_lock()