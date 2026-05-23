'''
Main search logic for RAG
- Builds a search query that combines recent conversation history with the current user input
- Runs a similarity search to find contextually relevant chunks
- Runs a keyword search to find chunks that literally match keywords in the user input
- Merges and deduplicates results from both searches, giving priority to similarity search results
'''
from config import DEBUG

def get_search_k(user_input):
    """
    Determine how many RAG results to return based on the length of the user input

    user_input: the current question or message from the user
    returns an integer k for how many RAG results to return
    """
    word_count = len(user_input.split())
    if word_count > 8:
        return 8
    return 5


def build_search_query(user_input, conversation_history, lookback=4):
    """
    Combine recent conversation context with current question so RAG understands the context of what it is looking for
    
    user_input: the current question or message from the user
    conversation_history: list of previous messages in the conversation
    lookback: how many recent messages to include in the query

    returns a string that combines recent conversation history with the current user input, formatted for RAG searching
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

def search_rag(rag, user_input, conversation_history):
    """
    Run similarity + keyword search and merge results, giving priority to similarity search results

    rag: the full rag object from rag.py, which has methods for both similarity and keyword search
    user_input: the current question or message from the user
    conversation_history: list of previous messages in the conversation

    returns a list of relevant chunks from the RAG search, merged from similarity and keyword search results with deduplication
    """
    k = get_search_k(user_input)
    search_query = build_search_query(user_input, conversation_history)

    # similarity search - finds semantically related content
    similarity_chunks = rag.search(search_query, top_k=k)
    # keyword search - finds chunks that literally contain the keywords from the query
    keyword_chunks = rag.search_by_keyword(user_input)[:10]
    if DEBUG:
        print(f"[DEBUG] Keyword search → {len(keyword_chunks)} chunks")

    # merge both, deduplicate by text, similarity results first for priority
    seen = {c["text"] for c in similarity_chunks}
    for kc in keyword_chunks:
        if kc["text"] not in seen: # add keyword search results that aren't already in similarity results
            similarity_chunks.append(kc)
            seen.add(kc["text"])

    return similarity_chunks