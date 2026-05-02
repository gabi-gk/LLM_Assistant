import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import DEBUG, INFORMATION_DIR, LOGS_DIR, CHROMA_DIR, K_DEFAULT
from core.chunking import Chunker

'''
The chunks are the data from the file the model is accessing
Mini model vectorisies the chunks
dbchrome stores them and later looks for ones that match current query context

'''

class RAG:
    def __init__(self):
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
        Index a single file into the vector database
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
        Index all supported files in a directory.
        """
        supported = {".py", ".txt", ".md", ".json", ".pdf"}
        path = Path(directory)

        if not path.exists():
            os.makedirs(path)
            print(f"[RAG] Created directory {directory} — add files here to index them.")
            return

        files = [f for f in path.rglob("*") if f.suffix in supported]

        if not files:
            print(f"[RAG] No supported files found in {directory}")
            return

        for f in files:
            self.index_file(str(f))

    def index_all(self):
        """
        Index both the notes folder and saved conversation logs
        """
        print("[RAG] Indexing knowledge base...")
        self.index_directory(INFORMATION_DIR)
        self.index_directory(LOGS_DIR)
        print(f"[RAG] Indexing complete. {self.collection.count()} total chunks.\n")

    def search(self, query, top_k=K_DEFAULT):
        """
        Convert query to a vector, find the closest chunks in the database.
        Returns a list of the most relevant text chunks.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # distance is 0-2 with cosine, convert to 0-1 similarity score
            similarity = 1 - (dist / 2)

            # boost personal txt and md notes 
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
        Split query into meaningful words, find chunks containing most of them
        """
        stop_words = {
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

        all_chunks = self.collection.get(include=["documents", "metadatas"])

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

    def format_context(self, chunks):
        """
        Format retrieved chunks into a string to inject into the prompt
        """
        if not chunks:
            return None

        lines = ["[Retrieved context from your files:]"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n--- Source {i}: {chunk['name']} ({chunk['type']}, relevance: {chunk['similarity']}) ---")
            lines.append(chunk["text"])

        return "\n".join(lines)