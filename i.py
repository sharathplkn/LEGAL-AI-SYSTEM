"""
Vector Database Builder — BNS / BNSS / BSA
Compatible with installed package versions:
    langchain-community==0.0.36
    langchain-huggingface==1.2.1
    langchain-text-splitters==0.0.2

Run this ONCE before starting a.py:
    python3 i.py
"""

import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── PDF files (lowercase filenames as found in your folder) ──────────────────
PDF_FILES = {
    "BNS":  "bns.pdf",
    "BNSS": "bnss.pdf",
    "BSA":  "bsa.pdf",
}

VECTOR_DB_DIR = "vector_db"

# ── Load and chunk a single PDF ───────────────────────────────────────────────
def load_pdf(law_name, pdf_path):
    print(f"  Loading {law_name} from {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    # Tag every page with the law name
    for page in pages:
        page.metadata["law"]    = law_name
        page.metadata["source"] = pdf_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Tag each chunk with its section number if detectable
    section_re = re.compile(r'\b(\d+[A-Z]?)\.\s')
    for chunk in chunks:
        m = section_re.search(chunk.page_content)
        chunk.metadata["section"] = m.group(1) if m else "Unknown"

    print(f"  ✓ {law_name}: {len(pages)} pages → {len(chunks)} chunks")
    return chunks


# ── Main builder ──────────────────────────────────────────────────────────────
def build():
    print("=" * 55)
    print("  Building Vector DB — BNS / BNSS / BSA")
    print("=" * 55)

    # Check all PDFs exist
    missing = [name for name, path in PDF_FILES.items() if not os.path.exists(path)]
    if missing:
        print(f"\n✗ Missing PDFs: {missing}")
        print("  Make sure bns.pdf, bnss.pdf, bsa.pdf are in this folder.")
        return

    # Delete old DB
    if os.path.exists(VECTOR_DB_DIR):
        print(f"\nRemoving old vector_db ...")
        shutil.rmtree(VECTOR_DB_DIR)

    # Load all PDFs
    print("\nLoading PDFs ...")
    all_chunks = []
    for law_name, pdf_path in PDF_FILES.items():
        all_chunks.extend(load_pdf(law_name, pdf_path))

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Load embedding model
    print("\nLoading embedding model ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✓ Embeddings ready")

    # Build and save Chroma DB
    print(f"\nEmbedding chunks into ChromaDB ...")
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    db.persist()

    print(f"\n✓ Vector database saved to '{VECTOR_DB_DIR}'")
    print("  Now run: python3 a.py")
    print("=" * 55)


if __name__ == "__main__":
    build()

