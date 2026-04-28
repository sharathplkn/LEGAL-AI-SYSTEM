"""
Indian Legal Assistant — BNS / BNSS / BSA Edition
Claude.ai-style UI using FastAPI + HTML/CSS/JS
Powered by: Groq API + LLaMA 3.3 (llama-3.3-70b-versatile)

Features:
  - Hybrid retrieval: content-based section lookup + semantic search
  - Meta-query handler for "what is BNS" style questions
  - History-aware follow-ups (e.g. "109?" after a BNS query)
  - IPC -> BNS mapping for legacy section references
  - Strict grounding: hard refusal when DB has no relevant content
  - BNS offence severity classifier (from BNSS First Schedule)
  - Situation -> applicable sections analyzer (/analyze endpoint)

Run:
    python3 a.py
Then open: http://127.0.0.1:8000
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("groq_key", "").strip().strip("'\"")
GROQ_MODEL    = "llama-3.3-70b-versatile"
VECTOR_DB_DIR = "vector_db"


# ── IPC → BNS mapping ─────────────────────────────────────────────────────────
IPC_TO_BNS = {
    "302":  ("103", "BNS", "Murder"),
    "304":  ("105", "BNS", "Culpable homicide not amounting to murder"),
    "304B": ("80",  "BNS", "Dowry death"),
    "306":  ("108", "BNS", "Abetment of suicide"),
    "307":  ("109", "BNS", "Attempt to murder"),
    "323":  ("115", "BNS", "Voluntarily causing hurt"),
    "354":  ("74",  "BNS", "Assault on woman to outrage modesty"),
    "354A": ("75",  "BNS", "Sexual harassment"),
    "354B": ("76",  "BNS", "Assault with intent to disrobe"),
    "354C": ("77",  "BNS", "Voyeurism"),
    "354D": ("78",  "BNS", "Stalking"),
    "375":  ("63",  "BNS", "Rape – definition"),
    "376":  ("64",  "BNS", "Punishment for rape"),
    "379":  ("303", "BNS", "Theft"),
    "380":  ("305", "BNS", "Theft in dwelling house"),
    "392":  ("309", "BNS", "Robbery"),
    "395":  ("310", "BNS", "Dacoity"),
    "406":  ("316", "BNS", "Criminal breach of trust"),
    "420":  ("318", "BNS", "Cheating"),
    "498A": ("86",  "BNS", "Cruelty by husband or relatives"),
}


# ── BNS Offence Severity & Procedural Classification ──────────────────────────
# From the First Schedule of BNSS, 2023.
# (offence_name, max_punishment, cognizable, bailable, court)
BNS_SEVERITY = {
    "63":  ("Rape – definition",                       "Reference section",   True,  False, "Sessions"),
    "64":  ("Punishment for rape",                     "Rigorous 10y to life", True,  False, "Sessions"),
    "65":  ("Rape of minor / aggravated rape",         "20y to life / death",  True,  False, "Sessions"),
    "66":  ("Rape causing death or vegetative state",  "20y to life / death",  True,  False, "Sessions"),
    "74":  ("Assault to outrage modesty",              "1–5 years + fine",     True,  False, "Any Magistrate"),
    "75":  ("Sexual harassment",                       "Up to 3 years + fine", True,  True,  "Any Magistrate"),
    "76":  ("Assault with intent to disrobe",          "3–7 years + fine",     True,  False, "Any Magistrate"),
    "77":  ("Voyeurism",                               "1–7 years + fine",     True,  False, "Any Magistrate"),
    "78":  ("Stalking",                                "Up to 3y (1st), 5y (repeat)", True, False, "Any Magistrate"),
    "80":  ("Dowry death",                             "7 years to life",      True,  False, "Sessions"),
    "85":  ("Cruelty by husband / relatives",          "Up to 3 years + fine", True,  False, "Magistrate 1st Class"),
    "86":  ("Cruelty – definition",                    "Reference section",    True,  False, "Magistrate 1st Class"),
    "103": ("Murder",                                  "Death / life + fine",  True,  False, "Sessions"),
    "104": ("Murder by life-convict",                  "Death / life",         True,  False, "Sessions"),
    "105": ("Culpable homicide not amounting to murder","Life / up to 10y + fine", True, False, "Sessions"),
    "106": ("Death by negligence",                     "Up to 5 years + fine", True,  True,  "Magistrate 1st Class"),
    "108": ("Abetment of suicide",                     "Up to 10 years + fine",True,  False, "Sessions"),
    "109": ("Attempt to murder",                       "Up to 10y / life + fine", True, False, "Sessions"),
    "115": ("Voluntarily causing hurt",                "Up to 1 year + fine",  False, True,  "Any Magistrate"),
    "117": ("Grievous hurt",                           "Up to 7 years + fine", True,  False, "Any Magistrate"),
    "118": ("Grievous hurt by dangerous weapon",       "Up to 10y / life",     True,  False, "Sessions"),
    "124": ("Acid attack",                             "10y to life + fine",   True,  False, "Sessions"),
    "127": ("Wrongful restraint",                      "Up to 1 month + fine", False, True,  "Any Magistrate"),
    "137": ("Kidnapping",                              "Up to 7 years + fine", True,  False, "Any Magistrate"),
    "303": ("Theft",                                   "Up to 3 years + fine", True,  False, "Any Magistrate"),
    "305": ("Theft in dwelling house / transit",       "Up to 7 years + fine", True,  False, "Any Magistrate"),
    "309": ("Robbery",                                 "Up to 10y rigorous + fine", True, False, "Magistrate 1st Class"),
    "310": ("Dacoity",                                 "Up to 10y / life + fine", True, False, "Sessions"),
    "316": ("Criminal breach of trust",                "Up to 5 years + fine", True,  False, "Magistrate 1st Class"),
    "318": ("Cheating",                                "Up to 7 years + fine", True,  False, "Magistrate 1st Class"),
    "319": ("Cheating by personation",                 "Up to 3 years + fine", True,  True,  "Any Magistrate"),
    "111": ("Organised crime",                         "5y to death + heavy fine", True, False, "Sessions"),
    "113": ("Terrorist act",                           "5y to death",          True,  False, "Sessions"),
    "329": ("Criminal trespass / house-trespass",      "Up to 3 months / 1y",  False, True,  "Any Magistrate"),
    "331": ("House-breaking",                          "Up to 2y / 10y + fine",True,  False, "Magistrate 1st Class"),
    "351": ("Criminal intimidation",                   "Up to 2 years / 7y",   False, True,  "Any Magistrate"),
    "356": ("Defamation",                              "Up to 2 years / fine", False, True,  "Magistrate 1st Class"),
}


def get_severity_badge(section_num):
    data = BNS_SEVERITY.get(section_num.upper())
    if not data:
        return ""
    name, punishment, cognizable, bailable, court = data
    cog_icon  = "Cognizable" if cognizable else "Non-cognizable"
    bail_icon = "Bailable"   if bailable   else "Non-bailable"
    return (
        f"\n\n> **Offence profile** — {cog_icon} · {bail_icon} · "
        f"{court} · Punishment: {punishment}"
    )


def find_bns_sections_in_text(text):
    found = []
    seen = set()
    for match in re.finditer(
        r'(?:(?:BNS|bns)\s*(?:§|Section|section)?\s*|(?:Section|section)\s+)(\d+[A-Za-z]?)\b',
        text,
    ):
        num = match.group(1).upper()
        if num not in seen:
            seen.add(num)
            found.append(num)
    return found


BNS_KEYWORDS = [
    "bns", "bharatiya nyaya sanhita", "murder", "theft", "rape", "stalking",
    "dowry", "cheating", "robbery", "assault", "hurt", "abetment",
    "culpable homicide", "cruelty", "husband", "voyeurism", "modesty",
    "criminal breach", "dacoity", "sexual harassment", "acid attack",
    "organised crime", "terrorism", "mob lynching",
    "offence against body", "offence against property",
]

BNSS_KEYWORDS = [
    "bnss", "bharatiya nagarik", "fir", "bail", "arrest", "trial",
    "charge sheet", "chargesheet", "cognizable", "investigation", "remand",
    "summons", "warrant", "police", "magistrate", "sessions court",
    "acquittal", "conviction", "appeal", "revision",
    "first information report", "custody", "anticipatory bail",
]

BSA_KEYWORDS = [
    "bsa", "bharatiya sakshya", "evidence", "confession", "witness",
    "electronic record", "digital evidence", "admissible", "admissibility",
    "burden of proof", "expert opinion", "dying declaration", "hearsay",
    "primary evidence", "secondary evidence", "estoppel", "presumption",
]


LAW_OVERVIEWS = {
    "BNS": (
        "## Bharatiya Nyaya Sanhita (BNS), 2023\n\n"
        "The **BNS** is India's new criminal code that **replaced the Indian Penal Code (IPC) 1860** "
        "with effect from **1st July 2024**.\n\n"
        "**Key facts:**\n"
        "- **Total sections:** 358 (IPC had 511)\n"
        "- **Purpose:** Defines criminal offences and their punishments\n"
        "- **Replaces:** Indian Penal Code, 1860\n\n"
        "**What it covers:**\n"
        "- Offences against the body (murder, hurt, kidnapping)\n"
        "- Offences against property (theft, robbery, dacoity, cheating)\n"
        "- Offences against women and children (rape, stalking, dowry death)\n"
        "- New offences: organised crime, terrorism, mob lynching\n\n"
        "Ask about any specific section (e.g. *'BNS Section 103'* or *'punishment for theft'*)."
    ),
    "BNSS": (
        "## Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023\n\n"
        "The **BNSS** is India's new criminal procedure code that **replaced the Code of Criminal "
        "Procedure (CrPC) 1973** with effect from **1st July 2024**.\n\n"
        "**Key facts:**\n"
        "- **Total sections:** 531 (CrPC had 484)\n"
        "- **Purpose:** Lays down the procedure for investigation, trial, and punishment\n"
        "- **Replaces:** Code of Criminal Procedure, 1973\n\n"
        "**What it covers:**\n"
        "- FIR, arrest, bail, and custody procedures\n"
        "- Investigation and charge-sheet rules\n"
        "- Trial by Magistrate and Sessions Courts\n"
        "- Appeals, revisions, and execution of sentences\n"
        "- Use of electronic evidence in trials\n\n"
        "Ask about specific provisions (e.g. *'how to file FIR under BNSS'*)."
    ),
    "BSA": (
        "## Bharatiya Sakshya Adhiniyam (BSA), 2023\n\n"
        "The **BSA** is India's new law of evidence that **replaced the Indian Evidence Act, 1872** "
        "with effect from **1st July 2024**.\n\n"
        "**Key facts:**\n"
        "- **Total sections:** 170 (Evidence Act had 167)\n"
        "- **Purpose:** Governs what evidence is admissible in Indian courts\n"
        "- **Replaces:** Indian Evidence Act, 1872\n\n"
        "**What it covers:**\n"
        "- Oral, documentary, and electronic evidence\n"
        "- Witness examination and cross-examination\n"
        "- Burden of proof and presumptions\n"
        "- Admissibility of digital records and forensic evidence\n"
        "- Dying declarations, confessions, and expert opinions\n\n"
        "Ask about specific provisions (e.g. *'electronic evidence under BSA'*)."
    ),
}

LEGACY_OVERVIEWS = {
    "ipc": (
        "**IPC** stood for the **Indian Penal Code, 1860** — repealed and replaced by "
        "**BNS (Bharatiya Nyaya Sanhita, 2023)** on 1st July 2024. "
        "Ask me about any specific IPC section and I'll show you its BNS equivalent."
    ),
    "crpc": (
        "**CrPC** stood for the **Code of Criminal Procedure, 1973** — repealed and replaced by "
        "**BNSS (Bharatiya Nagarik Suraksha Sanhita, 2023)** on 1st July 2024."
    ),
    "evidence act": (
        "The **Indian Evidence Act, 1872** was repealed and replaced by "
        "**BSA (Bharatiya Sakshya Adhiniyam, 2023)** on 1st July 2024."
    ),
}

DISCLAIMER = (
    "\n\n---\n"
    "*For educational purposes only. "
    "Consult a qualified lawyer for legal advice.*"
)

NOT_FOUND_MESSAGE = (
    "I could not find this in my database. "
    "Please check [indiacode.nic.in](https://indiacode.nic.in) "
    "or rephrase with a specific section number or more detail."
)


class IndianLegalAssistant:
    def __init__(self):
        print("=" * 60)
        print("  Indian Legal Assistant — BNS / BNSS / BSA")
        print(f"  Model : {GROQ_MODEL} via Groq")
        print("=" * 60)

        print("\nLoading embedding model ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  ✓ Embeddings ready")

        print("Loading vector database ...")
        self.db = None
        try:
            db = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=self.embeddings,
            )
            count = db._collection.count()
            if count == 0:
                print(f"  ✗ Vector DB is EMPTY. Run: python3 i.py to rebuild")
            else:
                print(f"  ✓ Vector database loaded — {count:,} chunks")
                self.db = db
        except Exception as e:
            print(f"  ✗ Vector DB error: {e}")

        print(f"Connecting to Groq ({GROQ_MODEL}) ...")
        self.llm = None
        try:
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.1,
                max_tokens=1024,
            )
            llm.invoke("Say OK only.")
            print(f"  ✓ Groq / {GROQ_MODEL} connected")
            self.llm = llm
        except Exception as e:
            print(f"  ✗ Groq error: {e}")

        self.chat_history = []

        self.legal_keywords = [
            "bns", "bnss", "bsa",
            "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
            "bharatiya sakshya adhiniyam",
            "ipc", "crpc", "evidence act",
            "section", "law", "legal", "court", "crime", "punishment",
            "offence", "penalty", "fine", "imprisonment", "arrest", "bail",
            "murder", "theft", "robbery", "assault", "cheating", "rape",
            "harassment", "dowry", "divorce", "marriage", "property",
            "supreme court", "high court", "judgment", "advocate", "lawyer",
            "petition", "fir", "complaint", "cognizable", "bailable",
            "warrant", "charge sheet", "trial", "acquittal", "conviction",
            "evidence", "witness",
            "stalk", "stalking", "harass", "disturb", "follow",
            "threat", "threaten", "abuse", "molest",
            "blackmail", "cybercrime", "eve tease",
            "obscene", "modesty", "acid", "voyeur",
        ]

        print("\n✓ Legal Assistant is ready!")
        print("  Open http://127.0.0.1:8000\n")

    def detect_law(self, message):
        msg = message.lower()
        scores = {
            "BNS":  sum(1 for kw in BNS_KEYWORDS  if kw in msg),
            "BNSS": sum(1 for kw in BNSS_KEYWORDS if kw in msg),
            "BSA":  sum(1 for kw in BSA_KEYWORDS  if kw in msg),
        }
        if re.search(r'\bbnss\b', msg): return "BNSS"
        if re.search(r'\bbsa\b', msg):  return "BSA"
        if re.search(r'\bbns\b', msg):  return "BNS"
        if max(scores.values()) == 0:
            return None
        return max(scores, key=scores.get)

    def infer_law_from_history(self):
        for m in reversed(self.chat_history[-6:]):
            text = m["content"].lower()
            if re.search(r'\bbnss\b', text) or "nagarik" in text: return "BNSS"
            if re.search(r'\bbsa\b', text) or "sakshya" in text:  return "BSA"
            if re.search(r'\bbns\b', text) or "nyaya" in text:    return "BNS"
        return None

    def extract_section_number(self, message):
        msg = message.strip()
        m = re.search(r'(?:section|sec)\s*\.?\s*(\d+[A-Z]?)', msg, re.IGNORECASE)
        if m: return m.group(1).upper()
        m = re.search(r'\b(?:bns|bnss|bsa|ipc|crpc)\s*\.?\s*(\d+[A-Z]?)\b',
                      msg, re.IGNORECASE)
        if m: return m.group(1).upper()
        stripped = msg.rstrip('?.!,').strip()
        if re.fullmatch(r'\d+[A-Z]?', stripped, re.IGNORECASE):
            return stripped.upper()
        return None

    def _content_lookup(self, section_num, law_filter):
        if not self.db:
            return []
        try:
            result = self.db._collection.get(where={"law": law_filter})
        except Exception as e:
            print(f"  [RAG] Content lookup failed for {law_filter}: {e}")
            return []
        if not result or not result.get("ids"):
            return []

        docs_text = result["documents"]
        metas = result["metadatas"]
        escaped = re.escape(section_num)

        patterns = [
            (1, re.compile(rf'(?<!\d){escaped}\.\s+[A-Z]')),
            (1, re.compile(rf'(?<!\d){escaped}\.[\u2014\u2013\u2010\-]')),
            (2, re.compile(rf'(?<!\d){escaped}\.[A-Z]')),
        ]

        hits = []
        for i, text in enumerate(docs_text):
            best = None
            for tier, pattern in patterns:
                match = pattern.search(text)
                if match:
                    candidate = (tier, match.start(), text, metas[i])
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            if best is not None:
                hits.append(best)
        hits.sort(key=lambda t: (t[0], t[1]))

        print(f"  [RAG] content_lookup §{section_num} in {law_filter}: {len(hits)} hits")
        return [Document(page_content=text, metadata=meta)
                for _tier, _pos, text, meta in hits[:6]]

    def _semantic_search(self, message, law_filter):
        if law_filter:
            docs = self.db.similarity_search(message, k=8, filter={"law": law_filter})
            if not docs:
                docs = self.db.similarity_search(message, k=8)
            return docs
        return self.db.similarity_search(message, k=8)

    def smart_retrieve(self, message, override_section=None, override_law=None,
                       override_hint=None):
        if not self.db:
            return []
        section_num = override_section or self.extract_section_number(message)
        law_filter  = override_law or self.detect_law(message) or self.infer_law_from_history()

        retrieval_query = message
        is_short = len(message.split()) < 8
        has_anchor = section_num is not None or bool(
            re.search(r'\b(?:bns|bnss|bsa|ipc|crpc)\b', message.lower())
        )
        if is_short and not has_anchor and self.chat_history:
            prev_user_msgs = [m["content"] for m in self.chat_history if m["role"] == "user"]
            if prev_user_msgs:
                retrieval_query = f"{prev_user_msgs[-1]} {message}"
                print(f"  [RAG] Vague follow-up — augmented with prev msg")

        print(f"  [RAG] law={law_filter or 'ALL'} section={section_num or 'None'}")

        if section_num and law_filter:
            direct = self._content_lookup(section_num, law_filter)
            semantic_q = (f"{override_hint} Section {section_num}"
                          if override_hint else f"Section {section_num} {law_filter}")
            semantic = self.db.similarity_search(
                semantic_q, k=4, filter={"law": law_filter})
            seen, merged = set(), []
            for doc in direct + semantic:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            if merged:
                return merged[:8]

        if section_num and not law_filter:
            for candidate in ["BNS", "BNSS", "BSA"]:
                direct = self._content_lookup(section_num, candidate)
                if direct:
                    return direct[:6]

        return self._semantic_search(retrieval_query, law_filter)

    def is_legal_query(self, message):
        msg = message.lower().strip()
        if re.search(r'(section|sec|bns|bnss|bsa|ipc|crpc)[.\s]*(\d+[A-Z]?)', msg):
            return True
        if re.fullmatch(r'\d+[A-Z]?[\s?.!]*', msg):
            for m in reversed(self.chat_history[-4:]):
                if m["role"] == "assistant":
                    lower = m["content"].lower()
                    if any(k in lower for k in ["bns", "bnss", "bsa", "section"]):
                        return True
                    break
        return any(kw in msg for kw in self.legal_keywords)

    def is_greeting(self, message):
        msg = message.lower().strip().rstrip('?.!,')
        greetings = ["hello", "hi", "hey", "greetings", "good morning",
                     "good afternoon", "good evening", "namaste", "namaskar"]
        for g in greetings:
            if msg == g: return True
            if msg.startswith(g + " ") or msg.startswith(g + ","): return True
        return False

    def check_ipc_reference(self, message):
        msg = message.lower()
        match = re.search(
            r'ipc[.\s]*(?:section|sec)?[.\s]*(\d+[A-Z]?)'
            r'|(?:section|sec)[.\s]*(\d+[A-Z]?).*ipc', msg)
        if match:
            sec = (match.group(1) or match.group(2) or "").upper()
            if sec in IPC_TO_BNS:
                bns_sec, law, desc = IPC_TO_BNS[sec]
                return (
                    f"\n\n> **IPC → New Law:** Old IPC Section {sec} (*{desc}*) "
                    f"is now **{law} Section {bns_sec}** under the Bharatiya Nyaya Sanhita, 2023."
                )
        return ""

    def handle_meta_query(self, message):
        clean = re.sub(r'[^\w\s]', '', message.lower()).strip()
        clean = re.sub(r'\s+', ' ', clean)
        if clean in ("ipc", "what is ipc", "what was ipc"): return LEGACY_OVERVIEWS["ipc"]
        if clean in ("crpc", "what is crpc", "what was crpc"): return LEGACY_OVERVIEWS["crpc"]
        if clean in ("evidence act", "what is evidence act"): return LEGACY_OVERVIEWS["evidence act"]
        meta_patterns = [
            r'^(bns|bnss|bsa)$',
            r'^what\s+(?:is|are|was)\s+(?:the\s+)?(bns|bnss|bsa)$',
            r'^(?:tell me about|explain|describe)\s+(?:the\s+)?(bns|bnss|bsa)$',
            r'^(bns|bnss|bsa)\s+(?:kya hai|meaning|definition|explain|full form)$',
            r'^full\s+form\s+of\s+(bns|bnss|bsa)$',
            r'^what\s+does\s+(bns|bnss|bsa)\s+(?:stand for|mean)$',
        ]
        for pattern in meta_patterns:
            m = re.search(pattern, clean)
            if m:
                for grp in m.groups():
                    if grp and grp.upper() in ("BNS", "BNSS", "BSA"):
                        return LAW_OVERVIEWS[grp.upper()]
        return None

    def get_history_text(self):
        if not self.chat_history:
            return "None"
        lines = []
        for m in self.chat_history[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            if len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def remember(self, user_msg, ai_msg):
        self.chat_history.append({"role": "user",      "content": user_msg})
        self.chat_history.append({"role": "assistant", "content": ai_msg})

    def clear_memory(self):
        self.chat_history.clear()

    def analyze_situation(self, scenario):
        if not self.llm:
            return {"error": "LLM not connected. Check groq_key in .env"}
        if not self.db:
            return {"error": "Vector database not loaded. Run: python3 i.py"}

        evasion_terms = [
            "how to escape", "how to avoid arrest", "how to hide",
            "escape from police", "avoid arrest", "hide evidence",
            "destroy evidence", "bribe the", "flee the",
            "get away with",
        ]
        low = scenario.lower()
        if any(term in low for term in evasion_terms):
            return {
                "error": (
                    "I can't help with evading legal consequences. "
                    "If you are in a legal situation, please consult a qualified lawyer "
                    "who can advise you lawfully."
                )
            }

        try:
            docs = self.db.similarity_search(scenario, k=10, filter={"law": "BNS"})
        except Exception as e:
            return {"error": f"Retrieval failed: {e}"}

        if not docs:
            return {"error": "No relevant context found in database."}

        parts = []
        for i, doc in enumerate(docs[:8], 1):
            sec = doc.metadata.get("section", "?")
            parts.append(f"[Source {i} — BNS §{sec}]\n{doc.page_content[:500]}")
        context = "\n\n".join(parts)

        system_prompt = (
            "You are an Indian criminal law analyst. Given an incident description "
            "and legal context from the Bharatiya Nyaya Sanhita (BNS), 2023, "
            "identify the BNS sections most likely to apply.\n\n"
            "STRICT RULES:\n"
            "1. Only cite sections that appear in the provided Legal Context.\n"
            "2. NEVER invent section numbers. If unsure, omit.\n"
            "3. Output ONLY valid JSON — no prose before or after.\n"
            "4. Schema:\n"
            "{\n"
            '  "sections": [\n'
            '    {"section": "103", "name": "Murder", "confidence": "high|medium|low", '
            '"reasoning": "one short sentence"}\n'
            "  ],\n"
            '  "summary": "one paragraph describing likely legal classification",\n'
            '  "next_steps": ["step 1", "step 2", "step 3"]\n'
            "}\n"
            "5. Include 1-4 sections. Rank by confidence (high first).\n"
            "6. 'next_steps' should suggest lawful actions only "
            "(e.g., 'File an FIR at the nearest police station', 'Consult a lawyer', "
            "'Preserve evidence such as photos or messages')."
        )

        user_prompt = (
            f"Incident description:\n{scenario}\n\n"
            f"Legal Context (BNS only):\n{context}\n\n"
            "Analyse the incident and output the JSON as specified. "
            "Only cite sections present in the Legal Context above."
        )

        try:
            result = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ])
            raw = result.content.strip()
        except Exception as e:
            return {"error": f"LLM call failed: {e}"}

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return {"error": "Model did not return JSON.", "raw": raw[:300]}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "raw": raw[:300]}

        enriched = []
        for s in parsed.get("sections", []):
            sec_num = str(s.get("section", "")).strip().upper()
            severity = BNS_SEVERITY.get(sec_num)
            item = {
                "section": sec_num,
                "name": s.get("name", ""),
                "confidence": s.get("confidence", "unknown"),
                "reasoning": s.get("reasoning", ""),
            }
            if severity:
                _, punishment, cognizable, bailable, court = severity
                item["punishment"] = punishment
                item["cognizable"] = cognizable
                item["bailable"]   = bailable
                item["court"]      = court
            enriched.append(item)

        return {
            "sections": enriched,
            "summary": parsed.get("summary", ""),
            "next_steps": parsed.get("next_steps", []),
        }

    def generate_response(self, message):
        if not self.llm:
            return "❌ Groq LLM not connected. Check your groq_key in .env"

        if self.is_greeting(message):
            resp = (
                "Namaste! I'm your **Indian Legal Assistant**, powered by LLaMA 3.3 (70B).\n\n"
                "I cover India's new criminal laws effective 1st July 2024:\n\n"
                "- **BNS** – Bharatiya Nyaya Sanhita, 2023 *(replaces IPC 1860)*\n"
                "- **BNSS** – Bharatiya Nagarik Suraksha Sanhita, 2023 *(replaces CrPC 1973)*\n"
                "- **BSA** – Bharatiya Sakshya Adhiniyam, 2023 *(replaces Evidence Act 1872)*\n\n"
                "Ask about any section, offence, or punishment. "
                "You can also describe a situation and I'll identify the relevant law."
            )
            self.remember(message, resp)
            return resp

        meta = self.handle_meta_query(message)
        if meta is not None:
            resp = meta + DISCLAIMER
            self.remember(message, resp)
            return resp

        is_legal  = self.is_legal_query(message)
        retrieved_law = None
        ipc_note  = self.check_ipc_reference(message)
        section_num = self.extract_section_number(message)

        ipc_redirect_section = None
        ipc_redirect_law = None
        ipc_redirect_desc = None
        if re.search(r'\bipc\b', message, re.IGNORECASE):
            for num in re.findall(r'\b(\d+[A-Za-z]?)\b', message):
                if num.upper() in IPC_TO_BNS:
                    new_sec, new_law, desc = IPC_TO_BNS[num.upper()]
                    ipc_redirect_section = new_sec
                    ipc_redirect_law = new_law
                    ipc_redirect_desc = desc
                    print(f"  [REDIRECT] IPC {num.upper()} -> {new_law} §{new_sec}")
                    break

        if is_legal and self.db:
            try:
                docs = self.smart_retrieve(
                    message,
                    override_section=ipc_redirect_section,
                    override_law=ipc_redirect_law,
                    override_hint=ipc_redirect_desc,
                )
            except Exception as e:
                print(f"Retrieval error: {e}")
                docs = []

            if not docs:
                resp = NOT_FOUND_MESSAGE + (ipc_note or "") + DISCLAIMER
                self.remember(message, resp)
                return resp

            parts = []
            for i, doc in enumerate(docs[:5], 1):
                law = doc.metadata.get("law", "Indian Law")
                sec = doc.metadata.get("section", "?")
                if retrieved_law is None:
                    retrieved_law = law
                parts.append(
                    f"[Source {i} — {law}, Section {sec}]\n"
                    f"{doc.page_content[:800]}"
                )
            context = "\n\n".join(parts)

            system_prompt = (
                "You are an expert Indian criminal law assistant covering:\n"
                "- Bharatiya Nyaya Sanhita (BNS), 2023 — replaces IPC 1860\n"
                "- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 — replaces CrPC 1973\n"
                "- Bharatiya Sakshya Adhiniyam (BSA), 2023 — replaces Evidence Act 1872\n\n"
                "STRICT RULES:\n"
                "1. Answer ONLY using the Legal Context provided.\n"
                "2. If the exact answer is NOT in the context, reply with EXACTLY: "
                f"\"{NOT_FOUND_MESSAGE}\"\n"
                "3. NEVER invent section numbers or punishments.\n"
                "4. NEVER reference IPC, CrPC, Evidence Act, BSF, CRPF, or any law "
                "outside BNS/BNSS/BSA unless the context mentions it.\n"
                "5. Always cite the exact law (BNS/BNSS/BSA) and section number from the context.\n"
                "6. Quote the punishment text verbatim from the context.\n"
                "7. Format: **Section X of <Law>** → brief plain-English explanation "
                "→ verbatim punishment quote."
            )

            effective_section = ipc_redirect_section or section_num

            if effective_section:
                redirect_hint = ""
                if ipc_redirect_section:
                    redirect_hint = (
                        f"\n\nIMPORTANT: The user asked about an old IPC section. "
                        f"Answer about the NEW law — {ipc_redirect_law} Section "
                        f"{ipc_redirect_section} — which replaced it. "
                        f"Do NOT describe the IPC section itself."
                    )
                user_prompt = (
                    f"Chat History:\n{self.get_history_text()}\n\n"
                    f"Legal Context:\n{context}\n\n"
                    f"Question: {message}\n\n"
                    f"The user is asking about Section {effective_section}. "
                    f"Find the chunk in the Legal Context containing "
                    f"'{effective_section}.' followed by a title. "
                    f"Quote the section title and its punishment verbatim. "
                    f"If Section {effective_section} is genuinely not in the context, "
                    f"respond with the exact refusal message from rule 2."
                    f"{redirect_hint}"
                )
            else:
                user_prompt = (
                    f"Chat History:\n{self.get_history_text()}\n\n"
                    f"Legal Context:\n{context}\n\n"
                    f"Question: {message}\n\n"
                    "Answer strictly from the Legal Context above. "
                    "Do not use any outside knowledge."
                )
        else:
            system_prompt = (
                "You are an expert Indian Legal Assistant for BNS, BNSS and BSA. "
                "When a user describes a situation, identify if any Indian law applies "
                "and explain the relevant BNS/BNSS/BSA section, offence, and punishment. "
                "If you are uncertain about a specific section number, do NOT guess — "
                "instead describe the offence generally and advise the user to ask about "
                "a specific section. NEVER cite IPC, CrPC, BSF, or CRPF."
            )
            user_prompt = (
                f"Chat History:\n{self.get_history_text()}\n\n"
                f"User: {message}\n\n"
                "Identify if this describes a legal situation under BNS/BNSS/BSA "
                "and explain the relevant law, section, and punishment if applicable. "
                "If not a legal matter, respond helpfully in 1-2 sentences."
            )

        try:
            result = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ])
            response = result.content.strip()
        except Exception as e:
            return f"❌ Groq API error: {e}"

        if ipc_note:
            response += ipc_note

        if is_legal and retrieved_law == "BNS":
            sections_in_response = find_bns_sections_in_text(response)
            badges_added = 0
            for sec in sections_in_response:
                if badges_added >= 2:
                    break
                badge = get_severity_badge(sec)
                if badge:
                    response += badge
                    badges_added += 1

        if is_legal:
            response += DISCLAIMER

        self.remember(message, response)
        return response


assistant = IndianLegalAssistant()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class AnalyzeRequest(BaseModel):
    scenario: str


@app.post("/chat")
async def chat(req: ChatRequest):
    response = assistant.generate_response(req.message)
    return JSONResponse({"response": response})


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.scenario or len(req.scenario.strip()) < 10:
        return JSONResponse(
            {"error": "Please describe the incident in more detail (at least 10 characters)."}
        )
    result = assistant.analyze_situation(req.scenario.strip())
    return JSONResponse(result)


@app.post("/clear")
async def clear():
    assistant.clear_memory()
    return JSONResponse({"status": "cleared"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Indian Legal Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:          #f5f1eb;
    --surface:     #ffffff;
    --surface-2:   #faf7f2;
    --border:      #e8e2d7;
    --border-2:    #d9d2c4;
    --accent:      #c96442;
    --accent-hover:#b0553a;
    --accent-soft: #f0e5dc;
    --text:        #2c2825;
    --text-2:      #6b655d;
    --text-3:      #958f85;
    --user-bg:     #f0e5dc;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --serif: 'Source Serif Pro', Georgia, serif;
  }

  [data-theme="dark"] {
    --bg:          #1a1917;
    --surface:     #262420;
    --surface-2:   #201e1b;
    --border:      #36322d;
    --border-2:    #48433c;
    --accent:      #e08a6a;
    --accent-hover:#eca689;
    --accent-soft: #3a2a22;
    --text:        #ebe6df;
    --text-2:      #a8a29a;
    --text-3:      #6f6a63;
    --user-bg:     #3a2a22;
  }

  html, body { height: 100%; }
  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    display: flex;
    overflow: hidden;
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    transition: background 0.2s ease, color 0.2s ease;
  }

  .sidebar {
    width: 260px;
    background: var(--surface-2);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 18px 12px;
    flex-shrink: 0;
    transition: transform 0.25s ease;
  }

  .logo {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 10px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
  }
  .logo-mark {
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-family: var(--serif); font-weight: 600; font-size: 16px;
  }
  .logo-text { font-family: var(--serif); font-size: 17px; font-weight: 600; color: var(--text); }
  .logo-sub  { font-size: 11px; color: var(--text-3); margin-top: 1px; }

  .new-chat-btn {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border-2);
    border-radius: 8px;
    padding: 9px 12px;
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: flex; align-items: center; gap: 8px;
    transition: all 0.15s;
    margin-bottom: 6px;
    width: 100%;
  }
  .new-chat-btn:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .new-chat-btn svg { width: 14px; height: 14px; }

  .sidebar-section {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 14px 10px 6px;
  }

  .quick-btn {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 7px 10px;
    color: var(--text-2);
    font-family: var(--sans);
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    transition: background 0.12s, color 0.12s;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .quick-btn:hover { background: var(--accent-soft); color: var(--text); }

  .law-tags { margin-top: auto; padding-top: 14px; border-top: 1px solid var(--border); }
  .law-tag {
    font-size: 11px;
    color: var(--text-2);
    padding: 6px 10px;
    display: flex; align-items: center; gap: 8px;
  }
  .law-tag b { color: var(--accent); font-weight: 600; font-size: 11px; min-width: 34px; }

  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

  .topbar {
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    gap: 12px;
    flex-shrink: 0;
  }
  .topbar-left { display: flex; align-items: center; gap: 10px; }
  .hamburger {
    display: none;
    background: transparent;
    border: 1px solid var(--border-2);
    color: var(--text-2);
    border-radius: 6px;
    width: 34px; height: 34px;
    cursor: pointer;
    align-items: center; justify-content: center;
  }
  .hamburger svg { width: 18px; height: 18px; }
  .topbar-title { font-family: var(--serif); font-size: 16px; font-weight: 600; }

  .tabs {
    display: flex;
    gap: 2px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 3px;
  }
  .tab-btn {
    background: transparent;
    border: none;
    color: var(--text-2);
    padding: 6px 14px;
    font-size: 13px;
    font-family: var(--sans);
    font-weight: 500;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .tab-btn.active { background: var(--accent); color: white; }
  .tab-btn:not(.active):hover { color: var(--text); background: var(--bg); }

  .topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .theme-toggle {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-2);
    border-radius: 8px;
    width: 34px; height: 34px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .theme-toggle:hover {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .theme-toggle svg {
    width: 16px;
    height: 16px;
    transition: opacity 0.2s;
  }
  .theme-toggle .moon-icon { display: none; }
  [data-theme="dark"] .theme-toggle .sun-icon { display: none; }
  [data-theme="dark"] .theme-toggle .moon-icon { display: block; }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 30px 0 20px;
  }
  .messages::-webkit-scrollbar { width: 8px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }

  .welcome {
    text-align: center;
    padding: 40px 24px;
    max-width: 620px;
    margin: 30px auto 0;
  }
  .welcome h2 {
    font-family: var(--serif);
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--text);
  }
  .welcome p {
    font-size: 15px;
    color: var(--text-2);
    line-height: 1.6;
    margin-bottom: 24px;
  }
  .welcome-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }
  .chip {
    background: var(--surface);
    border: 1px solid var(--border-2);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    color: var(--text-2);
    cursor: pointer;
    transition: all 0.15s;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }

  .message-row {
    display: flex;
    padding: 10px 24px;
    gap: 14px;
    max-width: 780px;
    margin: 0 auto;
    width: 100%;
    animation: fadeIn 0.25s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .avatar {
    width: 30px; height: 30px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--serif); font-weight: 600; font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .avatar.user { background: var(--user-bg); color: var(--accent); }
  .avatar.bot  { background: var(--accent); color: white; }

  .bubble {
    flex: 1;
    min-width: 0;
    word-wrap: break-word;
  }
  .bubble.user {
    background: var(--user-bg);
    border-radius: 12px;
    padding: 10px 14px;
    color: var(--text);
    font-size: 14.5px;
  }
  .bubble.bot {
    background: transparent;
    padding: 4px 0;
    color: var(--text);
    font-size: 15px;
    line-height: 1.7;
  }
  .bubble.bot p { margin-bottom: 12px; }
  .bubble.bot p:last-child { margin-bottom: 0; }
  .bubble.bot strong { font-weight: 600; color: var(--text); }
  .bubble.bot em { color: var(--text-2); font-style: italic; }
  .bubble.bot ul, .bubble.bot ol { padding-left: 22px; margin-bottom: 12px; }
  .bubble.bot li { margin-bottom: 4px; }
  .bubble.bot h2 {
    font-family: var(--serif);
    font-size: 20px;
    font-weight: 600;
    margin-top: 4px; margin-bottom: 10px;
  }
  .bubble.bot h3 { font-family: var(--serif); font-size: 16px; margin: 14px 0 6px; font-weight: 600; }
  .bubble.bot blockquote {
    border-left: 3px solid var(--accent);
    padding: 8px 14px;
    margin: 12px 0;
    background: var(--accent-soft);
    border-radius: 0 8px 8px 0;
    color: var(--text-2);
    font-size: 14px;
  }
  .bubble.bot blockquote strong { color: var(--accent); }
  .bubble.bot code {
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 13px;
    color: var(--accent);
  }
  .bubble.bot hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .bubble.bot a { color: var(--accent); text-decoration: underline; }

  .typing {
    display: flex; gap: 4px; padding: 10px 0; align-items: center;
  }
  .typing span {
    width: 7px; height: 7px;
    background: var(--text-3);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce {
    0%,80%,100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-5px); opacity: 1; }
  }

  .input-area {
    padding: 14px 24px 20px;
    background: var(--bg);
    flex-shrink: 0;
  }
  .input-wrapper {
    max-width: 780px;
    margin: 0 auto;
    background: var(--surface);
    border: 1px solid var(--border-2);
    border-radius: 12px;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    padding: 10px 12px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .input-wrapper:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  textarea {
    flex: 1;
    background: transparent;
    border: none; outline: none;
    color: var(--text);
    font-family: var(--sans);
    font-size: 14.5px;
    line-height: 1.5;
    resize: none;
    max-height: 160px;
    min-height: 22px;
    height: 22px;
    padding: 2px 0;
  }
  textarea::placeholder { color: var(--text-3); }
  .send-btn {
    background: var(--accent);
    border: none;
    border-radius: 8px;
    width: 32px; height: 32px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s;
    color: white;
  }
  .send-btn:hover { background: var(--accent-hover); }
  .send-btn:disabled { background: var(--border-2); cursor: not-allowed; }
  .send-btn svg { width: 16px; height: 16px; }

  .mic-btn {
    background: transparent;
    border: 1px solid var(--border-2);
    border-radius: 8px;
    width: 32px; height: 32px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: all 0.15s;
    color: var(--text-2);
  }
  .mic-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
  .mic-btn.listening {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
    animation: mic-pulse 1.3s infinite;
  }
  .mic-btn svg { width: 15px; height: 15px; }
  @keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(201, 100, 66, 0.5); }
    50% { box-shadow: 0 0 0 6px rgba(201, 100, 66, 0); }
  }

  .speaker-btn {
    background: transparent;
    border: none;
    color: var(--text-3);
    cursor: pointer;
    padding: 4px 6px;
    border-radius: 5px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11.5px;
    font-family: var(--sans);
    margin-top: 6px;
    transition: all 0.15s;
  }
  .speaker-btn:hover { background: var(--accent-soft); color: var(--accent); }
  .speaker-btn.speaking { color: var(--accent); background: var(--accent-soft); }
  .speaker-btn svg { width: 13px; height: 13px; }

  .input-footer {
    max-width: 780px;
    margin: 8px auto 0;
    font-size: 11.5px;
    color: var(--text-3);
    text-align: center;
  }

  .analyze-view {
    display: none;
    flex: 1;
    overflow-y: auto;
    padding: 40px 24px 30px;
  }
  .analyze-view.active { display: block; }
  .analyze-inner { max-width: 780px; margin: 0 auto; }
  .analyze-header { margin-bottom: 24px; }
  .analyze-header h2 {
    font-family: var(--serif);
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 6px;
  }
  .analyze-header p { font-size: 14px; color: var(--text-2); line-height: 1.6; }

  .analyze-input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border-2);
    border-radius: 10px;
    padding: 14px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 14.5px;
    line-height: 1.55;
    min-height: 110px;
    resize: vertical;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .analyze-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .analyze-actions {
    display: flex; gap: 10px; align-items: center;
    margin-top: 12px;
  }
  .analyze-btn {
    background: var(--accent);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
    font-family: var(--sans);
    cursor: pointer;
    transition: background 0.15s;
  }
  .analyze-btn:hover { background: var(--accent-hover); }
  .analyze-btn:disabled { background: var(--border-2); cursor: not-allowed; }
  .analyze-hint { font-size: 12px; color: var(--text-3); }

  .analyze-results { margin-top: 24px; }
  .summary-card {
    background: var(--accent-soft);
    border: 1px solid var(--border-2);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
  }

  .section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.15s;
  }
  .section-card:hover { border-color: var(--border-2); }
  .section-card-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; margin-bottom: 8px;
  }
  .section-title {
    font-family: var(--serif);
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
  }
  .section-title b { color: var(--accent); font-weight: 600; }
  .conf-pill {
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .conf-high   { background: #e3efe0; color: #2e6b24; }
  .conf-medium { background: #fdf4d8; color: #876617; }
  .conf-low    { background: #f6e3d5; color: #8a4a22; }

  [data-theme="dark"] .conf-high   { background: #1e3a1a; color: #a8d99a; }
  [data-theme="dark"] .conf-medium { background: #3a3015; color: #e8cc6a; }
  [data-theme="dark"] .conf-low    { background: #3a2519; color: #e5a37a; }

  .section-reason {
    font-size: 14px; color: var(--text-2); line-height: 1.55;
    margin-bottom: 10px;
  }
  .meta-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 10px;
  }
  .meta-pill {
    font-size: 11.5px;
    padding: 4px 9px;
    border-radius: 6px;
    background: var(--surface-2);
    color: var(--text-2);
    border: 1px solid var(--border);
  }
  .meta-pill.danger  { background: #fbeae4; color: #8a3a22; border-color: #f0cfc0; }
  .meta-pill.success { background: #e8f3e5; color: #2e6b24; border-color: #cfe3c8; }

  [data-theme="dark"] .meta-pill.danger  { background: #3a2519; color: #e5a37a; border-color: #5a3a2a; }
  [data-theme="dark"] .meta-pill.success { background: #1e3a1a; color: #a8d99a; border-color: #2e5a28; }

  .next-steps {
    margin-top: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  .next-steps h3 {
    font-family: var(--serif);
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--text);
  }
  .next-steps ol { padding-left: 20px; font-size: 14px; color: var(--text-2); }
  .next-steps li { margin-bottom: 6px; line-height: 1.55; }

  .error-card {
    background: #fbeae4;
    border: 1px solid #f0cfc0;
    color: #8a3a22;
    padding: 14px 16px;
    border-radius: 10px;
    font-size: 14px;
  }

  [data-theme="dark"] .error-card {
    background: #3a2519;
    border-color: #5a3a2a;
    color: #e5a37a;
  }

  .overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.3); z-index: 99;
  }
  .overlay.open { display: block; }

  @media (max-width: 720px) {
    .sidebar {
      position: fixed; top: 0; left: 0; bottom: 0;
      transform: translateX(-100%);
      z-index: 100; width: 240px;
    }
    .sidebar.open { transform: translateX(0); box-shadow: 4px 0 20px rgba(0,0,0,0.08); }
    .hamburger { display: flex; }
    .topbar { padding: 12px 16px; }
    .message-row, .input-area, .analyze-view, .input-footer { padding-left: 16px; padding-right: 16px; }
    .welcome h2 { font-size: 24px; }
    textarea { font-size: 16px; }
    .analyze-header h2 { font-size: 22px; }
    .tabs { padding: 2px; }
    .tab-btn { padding: 6px 11px; font-size: 12px; }
    .topbar-right { gap: 6px; }
    .theme-toggle { width: 30px; height: 30px; }
    .theme-toggle svg { width: 14px; height: 14px; }
  }
</style>
</head>
<body>

<div class="overlay" id="overlay" onclick="closeSidebar()"></div>

<aside class="sidebar" id="sidebar">
  <div class="logo">
    <div class="logo-mark">L</div>
    <div>
      <div class="logo-text">Legal Assistant</div>
      <div class="logo-sub">BNS · BNSS · BSA</div>
    </div>
  </div>

  <button class="new-chat-btn" onclick="newChat()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    New chat
  </button>

  <div class="sidebar-section">Quick prompts</div>
  <button class="quick-btn" onclick="sq('What is the punishment for murder under BNS?')">Murder — BNS</button>
  <button class="quick-btn" onclick="sq('What replaced IPC 302?')">IPC 302 → BNS?</button>
  <button class="quick-btn" onclick="sq('What is punishment for theft under BNS?')">Theft punishment</button>
  <button class="quick-btn" onclick="sq('What is stalking and its punishment under BNS?')">Stalking — BNS</button>
  <button class="quick-btn" onclick="sq('What is electronic evidence under BSA?')">Electronic evidence</button>
  <button class="quick-btn" onclick="sq('What is dowry death under BNS?')">Dowry death</button>

  <div class="law-tags">
    <div class="sidebar-section" style="padding-top:0;">Active laws</div>
    <div class="law-tag"><b>BNS</b> Bharatiya Nyaya Sanhita</div>
    <div class="law-tag"><b>BNSS</b> Nagarik Suraksha Sanhita</div>
    <div class="law-tag"><b>BSA</b> Sakshya Adhiniyam</div>
  </div>
</aside>

<main class="main">
  <div class="topbar">
    <div class="topbar-left">
      <button class="hamburger" onclick="openSidebar()" aria-label="Menu">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <div class="topbar-title">Indian Legal Assistant</div>
    </div>
    <div class="topbar-right">
      <div class="tabs">
        <button class="tab-btn active" id="tab-chat" onclick="switchTab('chat')">Chat</button>
        <button class="tab-btn" id="tab-analyze" onclick="switchTab('analyze')">Analyze</button>
      </div>
      <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle theme" title="Toggle light/dark mode">
        <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.05" y2="7.05"/><line x1="16.95" y1="16.95" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.05" y2="16.95"/><line x1="16.95" y1="7.05" x2="19.07" y2="4.93"/></svg>
        <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      </button>
    </div>
  </div>

  <div class="messages" id="messages">
    <div class="welcome" id="welcome">
      <h2>How can I help you today?</h2>
      <p>Ask about India's new criminal laws — BNS, BNSS, and BSA.<br/>
         Describe a situation or ask about a specific section.</p>
      <div class="welcome-chips">
        <div class="chip" onclick="sq('What is the punishment for murder under BNS?')">Murder — BNS</div>
        <div class="chip" onclick="sq('What replaced IPC 302?')">What replaced IPC 302?</div>
        <div class="chip" onclick="sq('What is stalking and its punishment under BNS?')">Stalking</div>
        <div class="chip" onclick="sq('What is electronic evidence under BSA?')">Electronic evidence</div>
        <div class="chip" onclick="sq('What is dowry death under BNS?')">Dowry death</div>
        <div class="chip" onclick="sq('What is the punishment for rape under BNS?')">Punishment for rape</div>
      </div>
    </div>
  </div>

  <div class="input-area" id="inputArea">
    <div class="input-wrapper">
      <textarea id="input"
        placeholder="Ask about BNS / BNSS / BSA or describe a situation..."
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"
        rows="1"></textarea>
      <button class="mic-btn" id="micBtn" onclick="toggleMic()" aria-label="Voice input" title="Voice input">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
      </button>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()" aria-label="Send">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </button>
    </div>
    <div class="input-footer">Educational only. Not a substitute for professional legal advice.</div>
  </div>

  <div class="analyze-view" id="analyzeView">
    <div class="analyze-inner">
      <div class="analyze-header">
        <h2>Situation analyzer</h2>
        <p>Describe what happened in plain English. The system will identify applicable BNS sections, their offence profile, and suggest lawful next steps.</p>
      </div>
      <textarea class="analyze-input" id="analyzeInput" placeholder="Example: My neighbour hit me with a rod during an argument and I'm injured. What sections apply?"></textarea>
      <div class="analyze-actions">
        <button class="analyze-btn" id="analyzeBtn" onclick="runAnalyze()">Analyze situation</button>
        <span class="analyze-hint">Grounded in BNS text. Output is suggestion only.</span>
      </div>
      <div class="analyze-results" id="analyzeResult"></div>
    </div>
  </div>
</main>

<script>
  let isLoading = false;

  // ── Speech Recognition (Speech-to-Text) ──
  let recognition = null;
  let isListening = false;

  function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      document.getElementById('micBtn').style.display = 'none';
      return;
    }
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-IN';  // Indian English; falls back gracefully

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      const input = document.getElementById('input');
      input.value = transcript;
      autoResize(input);
    };
    recognition.onend = () => {
      isListening = false;
      document.getElementById('micBtn').classList.remove('listening');
    };
    recognition.onerror = (e) => {
      isListening = false;
      document.getElementById('micBtn').classList.remove('listening');
      if (e.error !== 'no-speech' && e.error !== 'aborted') {
        console.warn('Speech recognition error:', e.error);
      }
    };
  }

  function toggleMic() {
    if (!recognition) {
      alert('Voice input is not supported in this browser. Try Chrome, Edge, or Safari.');
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      try {
        document.getElementById('input').value = '';
        recognition.start();
        isListening = true;
        document.getElementById('micBtn').classList.add('listening');
      } catch (e) {
        console.warn('Could not start recognition:', e);
      }
    }
  }

  // ── Speech Synthesis (Text-to-Speech) ──
  let currentUtterance = null;

  function stripMarkdown(text) {
    return text
      .replace(/```[\s\S]*?```/g, '')        // code blocks
      .replace(/`([^`]+)`/g, '$1')            // inline code
      .replace(/\*\*([^*]+)\*\*/g, '$1')      // bold
      .replace(/\*([^*]+)\*/g, '$1')          // italic
      .replace(/^#+\s*/gm, '')                // headers
      .replace(/^>\s*/gm, '')                 // blockquotes
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')// links
      .replace(/[\-\*]\s+/g, '')              // list bullets
      .replace(/\n{2,}/g, '. ')               // paragraph breaks
      .replace(/\n/g, ' ')                    // line breaks
      .replace(/\s+/g, ' ')
      .trim();
  }

  function toggleSpeak(btn, text) {
    if (!window.speechSynthesis) {
      alert('Text-to-speech is not supported in this browser.');
      return;
    }
    // If this button is currently speaking, stop it
    if (btn.classList.contains('speaking')) {
      window.speechSynthesis.cancel();
      btn.classList.remove('speaking');
      return;
    }
    // Cancel any other ongoing speech
    window.speechSynthesis.cancel();
    document.querySelectorAll('.speaker-btn.speaking').forEach(b => b.classList.remove('speaking'));

    const clean = stripMarkdown(text);
    if (!clean) return;
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = 'en-IN';
    u.rate = 1.0;
    u.pitch = 1.0;
    u.onend = () => btn.classList.remove('speaking');
    u.onerror = () => btn.classList.remove('speaking');
    currentUtterance = u;
    btn.classList.add('speaking');
    window.speechSynthesis.speak(u);
  }

  // Initialize speech recognition on page load
  initSpeechRecognition();

  // ── Theme Toggle (Light / Dark) ──
  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }

  function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const next = isDark ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('theme', next); } catch(e) {}
  }

  // Restore saved theme or follow system preference on first load
  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem('theme'); } catch(e) {}
    if (saved === 'dark' || saved === 'light') {
      applyTheme(saved);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      applyTheme('dark');
    }
  })();

  function openSidebar() {
    document.getElementById('sidebar').classList.add('open');
    document.getElementById('overlay').classList.add('open');
  }
  function closeSidebar() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('overlay').classList.remove('open');
  }
  function switchTab(tab) {
    const isChat = tab === 'chat';
    document.getElementById('tab-chat').classList.toggle('active', isChat);
    document.getElementById('tab-analyze').classList.toggle('active', !isChat);
    document.getElementById('messages').style.display = isChat ? 'block' : 'none';
    document.getElementById('inputArea').style.display = isChat ? 'block' : 'none';
    document.getElementById('analyzeView').classList.toggle('active', !isChat);
    if (!isChat) setTimeout(() => document.getElementById('analyzeInput').focus(), 100);
  }
  function autoResize(el) {
    el.style.height = '22px';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }
  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }
  function hideWelcome() {
    const w = document.getElementById('welcome');
    if (w) w.remove();
  }
  function escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function addMessage(role, text) {
    hideWelcome();
    const container = document.getElementById('messages');
    const row = document.createElement('div');
    row.className = 'message-row';
    const avatar = document.createElement('div');
    avatar.className = 'avatar ' + role;
    avatar.textContent = role === 'user' ? 'U' : 'L';
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + role;
    bubble.innerHTML = role === 'bot' ? marked.parse(text) : escapeHtml(text);

    // Add speaker button for bot messages
    if (role === 'bot' && window.speechSynthesis) {
      const speakBtn = document.createElement('button');
      speakBtn.className = 'speaker-btn';
      speakBtn.setAttribute('aria-label', 'Read aloud');
      speakBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg> Listen';
      speakBtn._text = text;
      speakBtn.onclick = function() { toggleSpeak(this, this._text); };
      bubble.appendChild(speakBtn);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }
  function addTyping() {
    hideWelcome();
    const container = document.getElementById('messages');
    const row = document.createElement('div');
    row.className = 'message-row';
    row.id = 'typing-row';
    row.innerHTML = '<div class="avatar bot">L</div><div class="bubble bot"><div class="typing"><span></span><span></span><span></span></div></div>';
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }
  function removeTyping() {
    const t = document.getElementById('typing-row');
    if (t) t.remove();
  }
  async function sendMessage() {
    if (isLoading) return;
    const input = document.getElementById('input');
    const msg = input.value.trim();
    if (!msg) return;
    // Stop any ongoing speech (listening or speaking) when sending
    if (isListening && recognition) recognition.stop();
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      document.querySelectorAll('.speaker-btn.speaking').forEach(b => b.classList.remove('speaking'));
    }
    input.value = '';
    input.style.height = '22px';
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    closeSidebar();
    addMessage('user', msg);
    addTyping();
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      const data = await res.json();
      removeTyping();
      addMessage('bot', data.response);
    } catch(e) {
      removeTyping();
      addMessage('bot', 'Connection error. Please try again.');
    }
    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }
  function sq(text) {
    document.getElementById('input').value = text;
    closeSidebar();
    sendMessage();
  }
  async function newChat() {
    await fetch('/clear', {method: 'POST'});
    closeSidebar();
    switchTab('chat');
    document.getElementById('messages').innerHTML = `
      <div class="welcome" id="welcome">
        <h2>How can I help you today?</h2>
        <p>Ask about India's new criminal laws — BNS, BNSS, and BSA.<br/>
           Describe a situation or ask about a specific section.</p>
        <div class="welcome-chips">
          <div class="chip" onclick="sq('What is the punishment for murder under BNS?')">Murder — BNS</div>
          <div class="chip" onclick="sq('What replaced IPC 302?')">What replaced IPC 302?</div>
          <div class="chip" onclick="sq('What is stalking and its punishment under BNS?')">Stalking</div>
          <div class="chip" onclick="sq('What is electronic evidence under BSA?')">Electronic evidence</div>
          <div class="chip" onclick="sq('What is dowry death under BNS?')">Dowry death</div>
          <div class="chip" onclick="sq('What is the punishment for rape under BNS?')">Punishment for rape</div>
        </div>
      </div>`;
  }

  async function runAnalyze() {
    const scenario = document.getElementById('analyzeInput').value.trim();
    if (scenario.length < 10) {
      alert('Please describe the incident in more detail (at least 10 characters).');
      return;
    }
    const btn = document.getElementById('analyzeBtn');
    const out = document.getElementById('analyzeResult');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    out.innerHTML = '<div class="section-card"><div class="typing"><span></span><span></span><span></span></div></div>';
    try {
      const r = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({scenario})
      });
      const d = await r.json();
      if (d.error) {
        out.innerHTML = '<div class="error-card">' + escapeHtml(d.error) + '</div>';
      } else {
        let html = '';
        if (d.summary) {
          html += '<div class="summary-card">' + escapeHtml(d.summary) + '</div>';
        }
        (d.sections || []).forEach(s => {
          const conf = (s.confidence || '').toLowerCase();
          const confClass = ['high','medium','low'].includes(conf) ? 'conf-' + conf : 'conf-low';
          let meta = '';
          if (s.cognizable === true)  meta += '<span class="meta-pill danger">Cognizable</span>';
          if (s.cognizable === false) meta += '<span class="meta-pill">Non-cognizable</span>';
          if (s.bailable === true)    meta += '<span class="meta-pill success">Bailable</span>';
          if (s.bailable === false)   meta += '<span class="meta-pill danger">Non-bailable</span>';
          if (s.court)       meta += '<span class="meta-pill">' + escapeHtml(s.court) + '</span>';
          if (s.punishment)  meta += '<span class="meta-pill">' + escapeHtml(s.punishment) + '</span>';
          html += '<div class="section-card">' +
            '<div class="section-card-head">' +
              '<div class="section-title"><b>BNS §' + escapeHtml(s.section) + '</b> — ' + escapeHtml(s.name || '') + '</div>' +
              '<span class="conf-pill ' + confClass + '">' + escapeHtml(s.confidence || 'n/a') + '</span>' +
            '</div>' +
            (s.reasoning ? '<div class="section-reason">' + escapeHtml(s.reasoning) + '</div>' : '') +
            (meta ? '<div class="meta-row">' + meta + '</div>' : '') +
          '</div>';
        });
        if (d.next_steps && d.next_steps.length) {
          html += '<div class="next-steps"><h3>Suggested next steps</h3><ol>';
          d.next_steps.forEach(s => html += '<li>' + escapeHtml(s) + '</li>');
          html += '</ol></div>';
        }
        html += '<p style="margin-top:20px;font-size:11.5px;color:var(--text-3);text-align:center;">Educational only. Consult a qualified lawyer before acting.</p>';
        out.innerHTML = html;
      }
    } catch(e) {
      out.innerHTML = '<div class="error-card">Connection error: ' + escapeHtml(e.message) + '</div>';
    }
    btn.disabled = false;
    btn.textContent = 'Analyze situation';
  }

  document.getElementById('input').focus();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)