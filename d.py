"""
diagnose.py — Diagnostic tool for Indian Legal Assistant RAG pipeline
Run: python3 diagnose.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

PDF_FILES = {
    "BNS":  "bns.pdf",
    "BNSS": "bnss.pdf",
    "BSA":  "bsa.pdf",
}
VECTOR_DB_DIR = "vector_db"

# Colors for terminal output
class C:
    OK    = "\033[92m✓\033[0m"
    FAIL  = "\033[91m✗\033[0m"
    WARN  = "\033[93m⚠\033[0m"
    INFO  = "\033[94mℹ\033[0m"
    BOLD  = "\033[1m"
    END   = "\033[0m"

def header(text):
    print(f"\n{C.BOLD}{'=' * 60}{C.END}")
    print(f"{C.BOLD}  {text}{C.END}")
    print(f"{C.BOLD}{'=' * 60}{C.END}")

def check(label, condition, detail=""):
    icon = C.OK if condition else C.FAIL
    print(f"  {icon} {label}" + (f"  {C.BOLD}→{C.END} {detail}" if detail else ""))
    return condition

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Files exist
# ─────────────────────────────────────────────────────────────────────────────
def test_files():
    header("TEST 1: PDF files on disk")
    all_ok = True
    for name, path in PDF_FILES.items():
        exists = os.path.exists(path)
        size = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "missing"
        if not check(f"{name}: {path}", exists, size):
            all_ok = False
    check(f"vector_db/ directory exists", os.path.isdir(VECTOR_DB_DIR),
          "found" if os.path.isdir(VECTOR_DB_DIR) else "NOT FOUND — run i.py first")
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: PDFs actually contain extractable text
# ─────────────────────────────────────────────────────────────────────────────
def test_pdf_extraction():
    header("TEST 2: PDF text extraction (catches scanned/image PDFs)")
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        check("langchain_community import", False, str(e))
        return False

    all_ok = True
    for name, path in PDF_FILES.items():
        if not os.path.exists(path):
            continue
        try:
            pages = PyPDFLoader(path).load()
            total_chars = sum(len(p.page_content) for p in pages)
            avg_chars = total_chars / max(len(pages), 1)

            ok = len(pages) > 0 and avg_chars > 100
            check(f"{name}: {len(pages)} pages, {total_chars:,} chars, avg {avg_chars:.0f}/page",
                  ok,
                  "looks good" if ok else "EMPTY — likely scanned PDF, needs OCR")

            if pages and len(pages) > 5:
                sample = pages[5].page_content[:200].replace("\n", " ").strip()
                print(f"      Sample (page 6): {C.BOLD}{sample[:150]}...{C.END}")

            if not ok:
                all_ok = False
        except Exception as e:
            check(f"{name} extraction", False, str(e))
            all_ok = False
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Vector DB loads and has chunks
# ─────────────────────────────────────────────────────────────────────────────
def test_vector_db():
    header("TEST 3: Vector database health")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError as e:
        check("langchain imports", False, str(e))
        return False, None, None

    try:
        print(f"  {C.INFO} Loading embedding model (first run may download ~90MB)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        check("Embedding model loaded", True)
    except Exception as e:
        check("Embedding model", False, str(e))
        return False, None, None

    try:
        db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        count = db._collection.count()
        ok = count > 0
        check(f"Chroma DB loaded with {count:,} chunks", ok,
              "healthy" if ok else "EMPTY — run i.py to rebuild")
        if not ok:
            return False, None, None
    except Exception as e:
        check("Chroma DB load", False, str(e))
        return False, None, None

    # Inspect metadata
    try:
        sample = db._collection.get(limit=20, include=["metadatas", "documents"])
        laws_found = {}
        for meta in sample["metadatas"]:
            law = meta.get("law", "MISSING")
            laws_found[law] = laws_found.get(law, 0) + 1

        print(f"\n  {C.INFO} Metadata sample (20 chunks):")
        for law, cnt in laws_found.items():
            print(f"      law={law}: {cnt} chunks")

        if "MISSING" in laws_found:
            check("All chunks have 'law' metadata", False,
                  "some chunks missing law tag — rebuild with updated i.py")
        else:
            check("All sampled chunks have 'law' metadata", True)

        # Check per-law counts across entire DB
        print(f"\n  {C.INFO} Full DB breakdown by law:")
        for law_name in PDF_FILES.keys():
            try:
                result = db._collection.get(where={"law": law_name}, limit=1)
                # Get actual count via query
                all_for_law = db._collection.get(where={"law": law_name})
                n = len(all_for_law["ids"])
                check(f"  {law_name}: {n:,} chunks", n > 0)
            except Exception as e:
                check(f"  {law_name} filter query", False, str(e))

    except Exception as e:
        check("Metadata inspection", False, str(e))

    return True, db, embeddings

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Retrieval actually returns relevant docs
# ─────────────────────────────────────────────────────────────────────────────
def test_retrieval(db):
    header("TEST 4: Similarity search returns relevant chunks")

    queries = [
        ("What is the punishment for murder under BNS?", "BNS"),
        ("What is theft under BNS?",                     "BNS"),
        ("How to file FIR under BNSS?",                  "BNSS"),
        ("What is electronic evidence under BSA?",       "BSA"),
    ]

    all_ok = True
    for q, expected_law in queries:
        print(f"\n  {C.INFO} Query: {C.BOLD}{q}{C.END}")

        # Unfiltered search
        try:
            docs = db.similarity_search(q, k=3)
            check(f"  Unfiltered search returned {len(docs)} docs", len(docs) > 0)
            for i, d in enumerate(docs, 1):
                law = d.metadata.get("law", "?")
                sec = d.metadata.get("section", "?")
                preview = d.page_content[:100].replace("\n", " ")
                print(f"      [{i}] {law} §{sec}: {preview}...")
        except Exception as e:
            check("  Unfiltered search", False, str(e))
            all_ok = False
            continue

        # Filtered search — plain filter
        try:
            docs_f = db.similarity_search(q, k=3, filter={"law": expected_law})
            ok = len(docs_f) > 0
            check(f"  Filter {{'law': '{expected_law}'}} returned {len(docs_f)} docs", ok)
            if not ok:
                all_ok = False
        except Exception as e:
            check(f"  Plain filter", False, str(e))

        # Filtered search — $eq operator
        try:
            docs_f2 = db.similarity_search(q, k=3, filter={"law": {"$eq": expected_law}})
            check(f"  Filter {{'law': {{'$eq': '{expected_law}'}}}} returned {len(docs_f2)} docs",
                  len(docs_f2) > 0)
        except Exception as e:
            check(f"  $eq filter", False, str(e))

    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Groq API connectivity
# ─────────────────────────────────────────────────────────────────────────────
def test_groq():
    header("TEST 5: Groq API connectivity")
    key = os.getenv("groq_key", "").strip().strip("'\"")
    if not check("groq_key in .env", bool(key),
                 f"length={len(key)}" if key else "MISSING — add to .env"):
        return False

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            groq_api_key=key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=50,
        )
        result = llm.invoke("Reply with just the word: OK")
        ok = "OK" in result.content.upper()
        check(f"Groq responded: '{result.content.strip()[:50]}'", ok)
        return ok
    except Exception as e:
        check("Groq API call", False, str(e))
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{C.BOLD}  Indian Legal Assistant — Diagnostic Tool{C.END}")

    t1 = test_files()
    t2 = test_pdf_extraction() if t1 else False
    t3_ok, db, _ = test_vector_db() if t1 else (False, None, None)
    t4 = test_retrieval(db) if t3_ok else False
    t5 = test_groq()

    header("SUMMARY")
    results = [
        ("PDF files present",        t1),
        ("PDF text extractable",     t2),
        ("Vector DB populated",      t3_ok),
        ("Retrieval works",          t4),
        ("Groq API reachable",       t5),
    ]
    for label, ok in results:
        check(label, ok)

    print()
    if not t1:
        print(f"{C.WARN} FIX: Place bns.pdf, bnss.pdf, bsa.pdf in this folder.")
    elif not t2:
        print(f"{C.WARN} FIX: PDFs are scanned images. Use OCR (pytesseract) or find text-based PDFs.")
    elif not t3_ok:
        print(f"{C.WARN} FIX: Run `python3 i.py` to build the vector database.")
    elif not t4:
        print(f"{C.WARN} FIX: Check metadata tagging in i.py — chunks may be missing 'law' field.")
    elif not t5:
        print(f"{C.WARN} FIX: Check your groq_key in .env file.")
    else:
        print(f"{C.OK} All checks passed. If the assistant still hallucinates, the issue is prompt/LLM behavior.")
    print()

if __name__ == "__main__":
    main()