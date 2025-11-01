from pypdf import PdfReader
import re
import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths for saved files
INDEX_PATH = "data/handbook_index.faiss"
DOCS_PATH = "data/handbook_docs.pkl"
EMBEDDER_PATH = "data/embedder.pkl"  # optional, for saving embedder config

def clean_text(text: str):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_index():
    """Build the index from the PDF (run once or when PDF updates)."""
    reader = PdfReader("data/harvardhandbook.pdf") 
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    cleaned_text = clean_text(text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    docs = splitter.create_documents([cleaned_text])
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode([d.page_content for d in docs], convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, 'wb') as f:
        pickle.dump([d.page_content for d in docs], f)  # Save just the text content
    
    print(f"Indexed and saved {index.ntotal} handbook chunks.")
    return index, docs, embedder

def load_index():
    """Load the index and documents from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("Index files not found. Building index...")
        return build_index()
    
    print("Loading index from disk...")
    index = faiss.read_index(INDEX_PATH)
    
    with open(DOCS_PATH, 'rb') as f:
        docs_texts = pickle.load(f)
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Loaded {index.ntotal} handbook chunks.")
    return index, docs_texts, embedder

if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index, docs_texts, embedder = load_index()
else:
    index, docs, embedder = build_index()
    docs_texts = [d.page_content for d in docs]

def retrieve(query, num_chunks = 2): #query should be a list of strings
    """Quickly retrieve relevant chunks using the saved index."""
    if isinstance(query, str):
        query = [query]
    output = []
    for q in query: 
        q_emb = embedder.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, num_chunks)
        output.extend([docs_texts[i] for i in I[0]])
    return output