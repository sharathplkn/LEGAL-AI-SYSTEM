"""
Vector Database Builder — BNS / BNSS / BSA
Run: python3 i.py
"""
import os
import re
import shutil
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_FILES = {
    "BNS":  "bns.pdf",
    "BNSS": "bnss.pdf",
    "BSA":  "bsa.pdf",
}
VECTOR_DB_DIR = "vector_db"


def load_pdf(law_name, pdf_path):
    print(f"  Loading {law_name} from {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"    → {len(pages)} pages loaded")

    for page in pages:
        page.metadata["law"] = law_name
        page.metadata["source"] = pdf_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Anchored regex: matches "103. Murder" at line start with capitalized word
    section_re = re.compile(r'(?m)^\s*(\d+[A-Z]?)\.\s+[A-Z]')
    tagged = 0
    for chunk in chunks:
        # Re-apply law tag (split_documents preserves metadata, but be safe)
        chunk.metadata["law"] = law_name
        m = section_re.search(chunk.page_content)
        if m:
            chunk.metadata["section"] = m.group(1)
            tagged += 1
        else:
            chunk.metadata["section"] = "Unknown"

    print(f"    → {len(chunks)} chunks ({tagged} with section numbers)")
    return chunks


def build():
    print("=" * 60)
    print("  Building Vector DB — BNS / BNSS / BSA")
    print("=" * 60)

    # Verify PDFs
    missing = [name for name, path in PDF_FILES.items() if not os.path.exists(path)]
    if missing:
        print(f"\n✗ Missing PDFs: {missing}")
        return

    # Nuke old DB
    if os.path.exists(VECTOR_DB_DIR):
        print(f"\nRemoving old {VECTOR_DB_DIR}/ ...")
        shutil.rmtree(VECTOR_DB_DIR)

    # Load all PDFs
    print("\nLoading and chunking PDFs ...")
    all_chunks = []
    for law_name, pdf_path in PDF_FILES.items():
        all_chunks.extend(load_pdf(law_name, pdf_path))

    print(f"\n  Total chunks across all laws: {len(all_chunks)}")

    # Sanity check
    if not all_chunks:
        print("\n✗ No chunks produced. Check PDFs.")
        return

    # Confirm metadata
    from collections import Counter
    law_counts = Counter(c.metadata.get("law", "MISSING") for c in all_chunks)
    print(f"  Breakdown: {dict(law_counts)}")

    # Load embedding model — use CPU to guarantee it works
    print("\nLoading embedding model (CPU) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},     # ← changed from "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  ✓ Embeddings ready")

    # Build Chroma DB (auto-persists when persist_directory is set)
    print(f"\nEmbedding {len(all_chunks)} chunks into Chroma ...")
    print("  (this takes 1-3 minutes on CPU)")
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )

    # Verify it actually wrote
    count = db._collection.count()
    print(f"\n  ✓ Chroma reports {count:,} chunks stored")

    if count == 0:
        print("\n✗ DB is still empty after build. Something is very wrong.")
        return

    # Quick retrieval smoke test
    print("\nSmoke test: searching for 'punishment for murder'...")
    test_docs = db.similarity_search("punishment for murder", k=2)
    for i, d in enumerate(test_docs, 1):
        law = d.metadata.get("law")
        sec = d.metadata.get("section")
        preview = d.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {law} §{sec}: {preview}...")

    print(f"\n✓ Vector database saved to '{VECTOR_DB_DIR}/'")
    print("  Now run: python3 a.py")
    print("=" * 60)


if __name__ == "__main__":
    build()