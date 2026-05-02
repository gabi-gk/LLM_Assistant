from config import DEBUG, COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT
from core.model import load_model, create_streamer
from core.history import compact_history, save_conversation, load_last_session, save_session_state
from core.rag import RAG
from agent.loop import run_agent
from tools.notifications import restore_reminders
import os

os.environ["PYTORCH_CUDA_ALLOC_CONsF"] = "expandable_segments:True"

def get_search_k(user_input):
    """
    Return higher k for queries that ask about specific named content vs general questions that just need a few relevant chunks
    """
    word_count = len(user_input.split())
    if word_count > 8:
        return 8
    return 5

def build_search_query(user_input, conversation_history, lookback=4):
    """
    Combine recent conversation context with current question so RAG understands the context of what it is looking for
    
    lookback — how many recent messages to include in the query
    """
    if not conversation_history:
        return user_input
    
    # grab the last few exchanges
    recent = conversation_history[-lookback:]
    
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "AI"
        # truncate long messages so the search query doesn't get huge
        content = msg["content"][:200]
        context_lines.append(f"{role}: {content}")
    
    # combine into one search string
    context_str = "\n".join(context_lines)
    return f"{context_str}\nUser: {user_input}"

def run_chat():

    conversation_history = []
    conversation_history = load_last_session()
    
    print("Loading model...")
    model, tokenizer = load_model()
    streamer = create_streamer(tokenizer)
    rag = RAG()
    rag.index_all()
    restore_reminders()

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
        if user_input.lower() == "clear":
            save_session_state("cleared")
            conversation_history = []
            print("History cleared.\n")
            continue
        
        # save history
        conversation_history.append({"role": "user", "content": user_input})
        
        # search avaliable data for additional information
        search_query = build_search_query(user_input, conversation_history)
        k = get_search_k(user_input)

        # similarity search — finds semantically related content
        similarity_chunks = rag.search(search_query, top_k=k)

        # keyword search — finds chunks that literally contain the topic
        keyword_chunks = rag.search_by_keyword(user_input)[:10]
        if DEBUG:
            print(f"[DEBUG] Keyword search → {len(keyword_chunks)} chunks")

        # merge both, deduplicate by text, similarity results first
        seen = {c["text"] for c in similarity_chunks}
        for kc in keyword_chunks:
            if kc["text"] not in seen:
                similarity_chunks.append(kc)
                seen.add(kc["text"])

        chunks = similarity_chunks

        if chunks:
            if DEBUG:
                print("\n[RAG found:]")
                for c in chunks:
                    print(f"  → {c['name']} ({c['type']}, relevance: {c['similarity']})")
                    print(f"    {c['text'][:100]}...")
                print()
            # inject context into the user message
            context = rag.format_context(chunks)
            augmented_history = conversation_history[:-1] + [{
                "role": "user",
                "content": f"{context}\n\nUser question: {user_input}"
            }]
        else:
            augmented_history = conversation_history


        # AI response
        print("AI: ", end="", flush=True)
        reply = run_agent(model, tokenizer, augmented_history, streamer)
        
        conversation_history.append({"role": "assistant", "content": reply})
        print()

if __name__ == "__main__":
    run_chat()