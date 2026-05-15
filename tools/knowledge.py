'''
Provides tools for searching the rag database by the model if the data is not in it's current context window
- search personal documents
- look through past conversation history 
'''
from pathlib import Path
from config import DEBUG, INFORMATION_DIR, LOGS_DIR
from datetime import datetime

SELF_MODEL_PATH = "data/information/marvin_self.md"


rag = None

def set_rag(rag_instance):
    """
    Register the RAG instance for use by knowledge tools
    
    rag_instance: the initialised RAG object from core/rag.py
    """
    global rag
    rag = rag_instance

def search_knowledge_base(query):
    """
    Search personal documents, notes and files for information if the users question indicates they are looking for specific information that the model doesn't know but might find

    query: the search query string
    returns: formatted string of relevant chunks from the knowledge base
    """
    if rag is None:
        return "[ERROR] Knowledge base not initialised"

    # search only information directory chunks
    similarity_chunks = rag.search(query)
    keyword_chunks = rag.search_by_keyword(query)[:5]

    info_path = Path(INFORMATION_DIR).resolve() # fix path to information dir to resolve any relative path issues

    # filter to information dir only
    similarity_chunks = [
        c for c in similarity_chunks
        if info_path in Path(c["source"]).resolve().parents
        or Path(c["source"]).resolve().parent == info_path
    ]
    keyword_chunks = [
        c for c in keyword_chunks
        if info_path in Path(c["source"]).resolve().parents
        or Path(c["source"]).resolve().parent == info_path
    ]

    # merge and deduplicate similarity and keyword results to preserve order
    seen = {c["text"] for c in similarity_chunks}
    for kc in keyword_chunks:
        if kc["text"] not in seen:
            similarity_chunks.append(kc)
            seen.add(kc["text"])

    if not similarity_chunks:
        return "[INFO] No relevant information found in knowledge base"

    if DEBUG:
        print(f"[DEBUG] search_knowledge_base: {len(similarity_chunks)} chunks found")

    return rag.format_context(similarity_chunks)


def search_conversation_logs(query):
    """
    Search past conversation history for information for previously discussed data

    query: the search query string
    returns: formatted string of relevant chunks from past conversations
    """
    if rag is None:
        return "[ERROR] Knowledge base not initialised"

    similarity_chunks = rag.search(query) # search all chunks using contextual similarity
    keyword_chunks = rag.search_by_keyword(query)[:5] # also search by keyword and take top 5 to increase chances of finding relevant info in logs

    logs_path = Path(LOGS_DIR).resolve() # fix path to logs dir to resolve any relative path issues

    # filter to logs directory only to prioritise conversation logs and avoid mixing with knowledge base results
    similarity_chunks = [
        c for c in similarity_chunks
        if logs_path in Path(c["source"]).resolve().parents
        or Path(c["source"]).resolve().parent == logs_path
    ]
    keyword_chunks = [
        c for c in keyword_chunks
        if logs_path in Path(c["source"]).resolve().parents
        or Path(c["source"]).resolve().parent == logs_path
    ]

    # merge and deduplicate similarity and keyword results to preserve order
    seen = {c["text"] for c in similarity_chunks}
    for kc in keyword_chunks:
        if kc["text"] not in seen:
            similarity_chunks.append(kc)
            seen.add(kc["text"])

    if not similarity_chunks:
        return "[INFO] No relevant information found in conversation logs"

    if DEBUG:
        print(f"[DEBUG] search_conversation_logs: {len(similarity_chunks)} chunks found")

    return rag.format_context(similarity_chunks)

def update_self_model(observation):
    """
    Append a new observation to Marvin's self model file.
    Called when Marvin learns something new about himself or Fenn
    worth remembering across sessions.

    observation: the fact or observation to save
    returns: success or error message
    """
    path = Path(SELF_MODEL_PATH)
    
    if not path.exists():
        return f"[ERROR] Self model not found at {SELF_MODEL_PATH}"
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    entry = f"\n- [{timestamp}] {observation}"
    
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
        if DEBUG:
            print(f"[SELF MODEL] Updated: {observation}")
        return f"[SUCCESS] Noted: {observation}"
    except Exception as e:
        return f"[ERROR] Could not update self model: {e}"