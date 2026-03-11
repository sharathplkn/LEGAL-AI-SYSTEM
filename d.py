"""
Vector DB Verification Script
Run: python3 check_db.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("=" * 55)
print("  Vector DB Verification")
print("=" * 55)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# Load DB
db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
collection = db._collection

# ── Basic stats ───────────────────────────────────────────────────────────────
total = collection.count()
print(f"\n📊 Total chunks in DB: {total}")

if total == 0:
    print("\n✗ Database is EMPTY — run python3 i.py first!")
    exit()

# ── Check law distribution ────────────────────────────────────────────────────
all_data = collection.get(include=["metadatas"])
metadatas = all_data["metadatas"]

law_counts = {}
section_counts = {}
for m in metadatas:
    law = m.get("law", "UNKNOWN")
    sec = m.get("section", "Unknown")
    law_counts[law]     = law_counts.get(law, 0) + 1
    section_counts[sec] = section_counts.get(sec, 0) + 1

print("\n📚 Chunks per law:")
for law, count in sorted(law_counts.items()):
    status = "✓" if count > 0 else "✗"
    print(f"  {status} {law}: {count} chunks")

missing = [law for law in ["BNS", "BNSS", "BSA"] if law not in law_counts]
if missing:
    print(f"\n✗ MISSING laws: {missing} — re-run python3 i.py")
else:
    print("\n✓ All 3 laws present (BNS, BNSS, BSA)")

# ── Sample content check ──────────────────────────────────────────────────────
print("\n📄 Sample chunks from each law:")
for law in ["BNS", "BNSS", "BSA"]:
    results = db.similarity_search(f"punishment section {law}", k=1,
                                   filter={"law": law})
    if results:
        doc = results[0]
        print(f"\n  [{law}] Section {doc.metadata.get('section','?')}:")
        print(f"  {doc.page_content[:200]}...")
    else:
        print(f"\n  [{law}] ✗ No results found!")

# ── Test queries ──────────────────────────────────────────────────────────────
print("\n🔍 Test queries:")
test_queries = [
    ("Murder punishment",        "BNS Section 103"),
    ("Theft punishment",         "BNS Section 303"),
    ("Electronic evidence",      "BSA Section 63"),
    ("FIR registration",         "BNSS Section 173"),
]

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
for query, expected in test_queries:
    docs = retriever.invoke(query)
    laws_found = [f"{d.metadata.get('law','?')} Sec {d.metadata.get('section','?')}"
                  for d in docs]
    print(f"  Query: '{query}'")
    print(f"  Expected: {expected}")
    print(f"  Got: {', '.join(laws_found)}")
    print()

print("=" * 55)
print("✓ Verification complete")
print("=" * 55)