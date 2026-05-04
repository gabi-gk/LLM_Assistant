'''
- Contains the Chunker class reponsible for splitting documents into chunks for RAG indexing
- Supports plain text, Python files (split by function/class), and PDFs (with adaptive strategies to handle different formatting)
'''

import ast
import re
from pathlib import Path
from pypdf import PdfReader
from config import DEBUG

class Chunker:
    '''
    Class responsible for splitting documents into chunks for RAG indexing
    '''
    @staticmethod
    def chunk_text(text, filepath, chunk_size=500, overlap=50):
        """
        Split plain text into overlapping chunks to ensure text isn't lost at boundaries

        text — the full text to split
        filepath — the source file path, used for metadata in the chunk dict
        chunk_size — characters per chunk
        overlap — characters shared between adjacent chunks
        returns a list of dicts with keys: text, source (filepath), type (text), name (filename)
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "source": filepath,
                "type": "text",
                "name": Path(filepath).name
            })
            start += chunk_size - overlap  # step forward with overlap

        return chunks

    @staticmethod
    def chunk_python_file(filepath):
        """
        Split a .py file into function/class per chunk
        Uses AST (abstract syntax tree) so it splits on the function's actual ending

        filepath — the path to the .py file
        returns a list of dicts with keys: text, source (filepath), type (function/class/file_header), name (function/class name or filename)
        """
        chunks = []
        source = Path(filepath).read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # if the file can't be parsed just treat it as plain text
            return Chunker.chunk_text(source, filepath)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # extract the raw source lines for this function/class
                start = node.lineno - 1
                end = node.end_lineno
                chunk_source = "\n".join(source.splitlines()[start:end])

                chunks.append({
                    "text": chunk_source,
                    "source": filepath,
                    "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                    "name": node.name
                })

        # Get the first 20 lines in case of any imports etc.
        top_lines = "\n".join(source.splitlines()[:20])
        chunks.append({
            "text": top_lines,
            "source": filepath,
            "type": "file_header",
            "name": Path(filepath).name,
        })

        return chunks
    
    @staticmethod
    def chunk_pdf(filepath):
        """
        Extract text from PDF using adaptive splitting — tries multiple strategies and picks the best result

        filepath — the path to the PDF file
        returns a list of dicts with keys: text, source (filepath), type (text), name (filename)
        """
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # strategy 1: double newline (well formatted PDFs)
        double_nl = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        
        # strategy 2: single newline (most PDFs)
        single_nl = [p.strip() for p in full_text.split("\n") if p.strip()]

        # pick strategy based on what produces reasonable paragraph lengths
        if len(double_nl) > 5 and sum(len(p) for p in double_nl) / len(double_nl) > 100:
            if DEBUG:
                print(f"[RAG] PDF strategy: double newline ({len(double_nl)} paragraphs)")
            paragraphs = double_nl
        elif len(single_nl) > 10:
            if DEBUG:
                print(f"[RAG] PDF strategy: single newline ({len(single_nl)} lines)")
            paragraphs = single_nl
        else:
            # fall back to treating whole text as one block and chunking by size
            if DEBUG:
                print(f"[RAG] PDF strategy: character fallback")
            return Chunker.chunk_text(full_text, filepath)

        # Filter out the table of contents
        paragraphs = [p for p in paragraphs if not re.search(r'(\.\s){3,}', p)]

        if DEBUG:
            print(f"[DEBUG] Lines after TOC filter: {len(paragraphs)}")

        # regroup lines into ~800 char chunks with overlap to ensure the AI can access the whole thing
        chunks = []
        current_chunk = ""
        overlap_buffer = ""
        current_section = "preamble"

        for para in paragraphs:
            # start a new chunk for a new heading
            if Chunker.check_heading(para) and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": filepath,
                    "type": "text",
                    "name": Path(filepath).name,
                    "section": current_section
                })
                overlap_buffer = current_chunk[-100:]
                current_section = para.strip()
                current_chunk = overlap_buffer + para + " "
                overlap_buffer = ""
            # if adding this paragraph keeps us under chunk_size, add it
            elif len(current_chunk) + len(para) < 800:
                current_chunk += para + " "
            else:
                # save current chunk and start a new one
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": filepath,
                        "type": "text",
                        "name": Path(filepath).name,
                        "section": current_section
                    })
                    # keep last 100 chars as overlap so context isn't lost at boundaries
                    overlap_buffer = current_chunk[-100:]
                current_chunk = overlap_buffer + para + " "
                overlap_buffer = ""

        # add the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": filepath,
                "type": "text",
                "name": Path(filepath).name,
                "section": current_section
            })

        if DEBUG:
            print(f"[DEBUG] Chunks created: {len(chunks)}")
            for i, c in enumerate(chunks[:3]):
                print(f"  chunk {i}: {len(c['text'])} chars — '{c['text'][:80]}'")

        return chunks
    
    @staticmethod
    def check_heading(line):
        """
        Detect section headings 

        line — a line of text to check
        returns True if the line looks like a heading, False otherwise
        """
        line = line.strip()
        if not line:
            return False
        # numbered sections: 
        if re.match(r'^\d+(\.\d+)*\s+\w', line):
            return True
        # short ALL CAPS lines 
        if line.isupper() and 3 < len(line) < 80:
            return True
        return False