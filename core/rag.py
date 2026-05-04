'''
Manages the Retrieval-Augmented Generation (RAG) system for the assistant
- indexes (prepares for future use and saves) supported file types (.py, .txt, .md, .json, .pdf) from a specified directory
- uses a small embedding model to convert text into vectors for efficient similarity search
- provides methods to index files, search for relevant chunks, and format them for inclusion in the prompt
- uses a vector database (ChromaDB) to store embeddings of text chunks from files
- retrieves them based on similarity to the user's query
'''
import os
import chromadb
from matplotlib import lines
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import DEBUG, INFORMATION_DIR, LOGS_DIR, CHROMA_DIR, K_DEFAULT
from core.chunking import Chunker

class RAG:
    '''
    Main class for managing the Retrieval-Augmented Generation system
    '''
    def __init__(self):
        '''
        Initialize the embedding model and vector database client
        '''
        print("[RAG] Loading embedding model...")
        # small NN that converts text into vectors
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # access the chroma database to search and store the vectors
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name="assistant_knowledge",
            metadata={"hnsw:space": "cosine"} # cosine similarity for text matching
        )
        print(f"[RAG] Ready. {self.collection.count()} chunks indexed.\n")

    def index_file(self, filepath):
        """
        Index a single file into the vector database, for later retrieval based on content similarity

        filepath: path to the file to be indexed
        """
        path = Path(filepath)

        # index based on filetype 
        if path.suffix == ".py":
            chunks = Chunker.chunk_python_file(filepath)
        elif path.suffix == ".pdf":
            chunks = Chunker.chunk_pdf(filepath)
        elif path.suffix in (".txt", ".md", ".json"):
            text = path.read_text(encoding="utf-8")
            chunks = Chunker.chunk_text(text, filepath)
            if DEBUG:
                print(f"[DEBUG] {path.name} produced {len(chunks)} chunks:")
                for c in chunks[:3]:
                    print(f"  '{c['text'][:100]}'")
        else:
            print(f"[RAG] Skipping unsupported file type: {filepath}")
            return
        
        if not chunks:  
            return

        # generate embeddings for all chunks in this file at once
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        # build unique IDs so re-indexing the same file doesn't create duplicates
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": c["source"],
            "type": c["type"],
            "name": c["name"],
            "section": c.get("section", "unknown")
        } for c in chunks]

        # upsert = insert if new, update if already exists
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"[RAG] Indexed {len(chunks)} chunks from {path.name}")

    def index_directory(self, directory):
        """
        Index all supported files in a directory by calling index_file on each separately

        directory: path to the directory to be indexed   
        """
        supported = {".py", ".txt", ".md", ".json", ".pdf"}
        path = Path(directory)

        if not path.exists(): # if the directory doesn't exist, create it and prompt user to add files
            os.makedirs(path)
            print(f"[RAG] Created directory {directory} — add files here to index them.")
            return

        files = [f for f in path.rglob("*") if f.suffix in supported] # find all supported files in this directory and subdirectories

        if not files:
            print(f"[RAG] No supported files found in {directory}")
            return

        for f in files: # index each file separately to track progress and avoid memory issues with large files
            self.index_file(str(f))

    def index_all(self):
        """
        Index both the user's notes folder and saved conversation logs
        """
        print("[RAG] Indexing knowledge base...")
        self.index_directory(INFORMATION_DIR)
        self.index_directory(LOGS_DIR)
        print(f"[RAG] Indexing complete. {self.collection.count()} total chunks.\n")

    def search(self, query, top_k=K_DEFAULT):
        """
        Convert query to a vector, find the closest chunks in the database

        query: the user's question or statement to find relevant context for
        top_k: how many relevant chunks to return at most (from config)
        Returns a list of the most relevant text chunks
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist() # convert the query into a vector using the same model as for indexing

        results = self.collection.query( # find the most similar chunks based on cosine similarity of the vectors
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"] 
        )

        chunks = []
        for doc, meta, dist in zip( # combine the retrieved documents, their metadata, and their distance scores
            results["documents"][0], # actual raw text that gets injected into the prompt as context
            results["metadatas"][0], # dictionary of extra information about the chunk, source file, type, name, section etc
            results["distances"][0] # distance score from the query vector and the chunk vector, lower = more similar
        ):
            # distance is 0-2 with cosine, convert to 0-1 similarity score for convinience
            similarity = 1 - (dist / 2) # now 1 is most similar and 0 is least

            # boost personal txt and md notes, as they are more likely to be relevant than code or pdf chunks
            source = meta.get("source", "")
            if any(source.endswith(ext) for ext in (".txt", ".md")):
                similarity = min(1.0, similarity + 0.1)

            # only return chunks above a relevance threshold so no irrelevant ones are included
            if similarity > 0.3:
                chunks.append({
                    "text": doc,
                    "source": meta["source"],
                    "type": meta["type"],
                    "name": meta["name"],
                    "similarity": round(similarity, 2)
                })

        return chunks

    def search_by_keyword(self, query):
        """
        Split query into meaningful words, find chunks containing most of them - useful when searching for specific names, titles or phrases

        query: the user's question or statement to find relevant context for
        Returns a list of the most relevant text chunks based on keyword matching
        """
        stop_words = { # common words to ignore in keyword search, as they don't add meaning
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'do', 'does', 'did', 'will', 'would', 'can',
            'could', 'should', 'may', 'might', 'what', 'where', 'when',
            'who', 'how', 'why', 'which', 'that', 'this', 'it', 'its',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'about', 'tell', 'me', 'my', 'your', 'say', 'says', 'said',
            'find', 'show', 'get', 'give', 'make', 'use', 'using',
            'and', 'or', 'but', 'not', 'no', 'so', 'if', 'then',
            'section', 'chapter', 'file', 'document', 'project', 'report'
        }
        
        # split into words, filter stop words, keep meaningful terms
        words = [
            w.strip('.,?!:;') for w in query.lower().split()
            if w.strip('.,?!:;') not in stop_words
            and len(w.strip('.,?!:;')) > 2
        ]
        
        if not words:
            return []

        all_chunks = self.collection.get(include=["documents", "metadatas"]) # keyword search doesn't use embeddings

        # check the chunks for how many query words they contain
        matches = []
        for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
            doc_lower = doc.lower()
            # count how many keywords appear in this chunk
            hits = sum(1 for word in words if word in doc_lower)
            # require at least half the keywords to match
            if hits >= max(1, len(words) // 2):
                matches.append({
                    "text": doc,
                    "source": meta["source"],
                    "type": meta["type"],
                    "name": meta["name"],
                    "section": meta.get("section", ""),
                    "similarity": round(hits / len(words), 2)
                })
    
        # sort by how many keywords matched
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def format_context(self, chunks, max_chars=3000):
        """
        Format retrieved chunks into a string to inject into the prompt

        chunks: list of relevant text chunks with metadata
        max_chars: maximum total characters to include in the formatted context, to avoid overwhelming the model
        returns: a formatted string combining the most relevant chunks for the model to use as context
        """
        if not chunks:
            return None

        lines = ["[Retrieved context from your files:]"]
        total_chars = 0

        for i, chunk in enumerate(chunks, 1): # combine the retrieved chunks into a single string with clear separators and source info, so the model can use it as context for answering
            if total_chars >= max_chars:
                lines.append(f"\n[Context cap reached at {max_chars} characters]")
                break
            lines.append(f"\nSource {i}: {chunk['name']} ({chunk['type']}, relevance: {chunk['similarity']})")
            lines.append(chunk["text"])
            total_chars += len(chunk["text"]) # estimate of the added characters including formatting, to keep track of the total context length

        return "\n".join(lines)