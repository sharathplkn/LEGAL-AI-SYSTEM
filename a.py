"""
Indian Legal Assistant — BNS / BNSS / BSA Edition
ChatGPT-style UI using FastAPI + HTML/CSS/JS
Powered by: Groq API + LLaMA 3.3 (llama-3.3-70b-versatile)

Install extra dependency:
    uv pip install fastapi uvicorn

Run:
    python3 a.py
Then open: http://127.0.0.1:8000
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("groq_key", "").strip().strip("'\"")
GROQ_MODEL   = "llama-3.3-70b-versatile"

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


class IndianLegalAssistant:
    def __init__(self):
        print("=" * 55)
        print("  Indian Legal Assistant — BNS / BNSS / BSA")
        print(f"  Model : {GROQ_MODEL} via Groq")
        print("=" * 55)

        print("\nLoading embedding model ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✓ Embeddings ready")

        print("Loading vector database ...")
        try:
            self.db = Chroma(
                persist_directory="vector_db",
                embedding_function=self.embeddings,
            )
            print("✓ Vector database loaded")
        except Exception as e:
            print(f"✗ Vector DB error: {e}")
            self.db = None

        self.retriever = (
            self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            if self.db else None
        )

        print(f"Connecting to Groq ({GROQ_MODEL}) ...")
        try:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.1,
                max_tokens=1024,
            )
            self.llm.invoke("Say OK only.")
            print(f"✓ Groq / {GROQ_MODEL} connected")
        except Exception as e:
            print(f"✗ Groq error: {e}")
            self.llm = None

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
            "threat", "threaten", "abuse", "molest", "reject",
            "force", "blackmail", "cybercrime", "eve tease",
            "obscene", "modesty", "acid", "voyeur",
        ]

        print("\n✓ Legal Assistant is ready!")
        print("  Open http://127.0.0.1:8000\n")

    def is_legal_query(self, message):
        msg = message.lower()
        if re.search(r'(section|sec|bns|bnss|bsa|ipc)[.\s]*(\d+[A-Z]?)', msg):
            return True
        return any(kw in msg for kw in self.legal_keywords)

    def check_ipc_reference(self, message):
        msg = message.lower()
        match = re.search(
            r'ipc[.\s]*(?:section|sec)?[.\s]*(\d+[A-Z]?)'
            r'|(?:section|sec)[.\s]*(\d+[A-Z]?).*ipc', msg
        )
        if match:
            sec = (match.group(1) or match.group(2) or "").upper()
            if sec in IPC_TO_BNS:
                bns_sec, law, desc = IPC_TO_BNS[sec]
                return (
                    f"\n\n> 📌 **IPC → New Law:** Old IPC Section {sec} (*{desc}*) "
                    f"is now **{law} Section {bns_sec}** under the Bharatiya Nyaya Sanhita, 2023."
                )
        return ""

    def get_history_text(self):
        if not self.chat_history:
            return "None"
        lines = []
        for m in self.chat_history[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def remember(self, user_msg, ai_msg):
        self.chat_history.append({"role": "user",      "content": user_msg})
        self.chat_history.append({"role": "assistant", "content": ai_msg})

    def clear_memory(self):
        self.chat_history.clear()

    def generate_response(self, message):
        if not self.llm:
            return "❌ Groq LLM not connected. Check your groq_key in .env"

        is_legal = self.is_legal_query(message)
        ipc_note = self.check_ipc_reference(message)

        greetings = ["hello", "hi", "hey", "greetings", "good morning",
                     "good afternoon", "good evening", "namaste"]
        if any(g in message.lower() for g in greetings):
            resp = (
                "Namaste! 🙏 I'm your **Indian Legal Assistant**, powered by LLaMA 3.3 (70B).\n\n"
                "I cover India's new criminal laws effective 1st July 2024:\n\n"
                "- 📘 **BNS** – Bharatiya Nyaya Sanhita, 2023 *(replaces IPC 1860)*\n"
                "- 📗 **BNSS** – Bharatiya Nagarik Suraksha Sanhita, 2023 *(replaces CrPC 1973)*\n"
                "- 📙 **BSA** – Bharatiya Sakshya Adhiniyam, 2023 *(replaces Evidence Act 1872)*\n\n"
                "Ask about any section, offence, or punishment. You can also describe a situation and I'll identify the relevant law!"
            )
            self.remember(message, resp)
            return resp

        if is_legal and self.retriever:
            try:
                docs = self.retriever.invoke(message)
                history_text = self.get_history_text()

                if docs:
                    parts = []
                    for i, doc in enumerate(docs[:4], 1):
                        law = doc.metadata.get("law", "Indian Law")
                        sec = doc.metadata.get("section", "?")
                        parts.append(f"[Source {i} — {law}, Section {sec}]\n{doc.page_content[:600]}")
                    context = "\n\n".join(parts)

                    system_prompt = """You are an expert Indian criminal law assistant covering:
- Bharatiya Nyaya Sanhita (BNS), 2023 — replaces IPC 1860
- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 — replaces CrPC 1973
- Bharatiya Sakshya Adhiniyam (BSA), 2023 — replaces Evidence Act 1872

STRICT RULES:
1. Answer ONLY from the Legal Context provided below.
2. If the exact section is NOT in the context, say: "I could not find this in my database. Please check indiacode.nic.in or consult a lawyer." Do NOT guess any section number.
3. NEVER hallucinate any section number or punishment.
4. Always cite the exact law (BNS/BNSS/BSA) and section number from context.
5. Quote punishments verbatim from the law text.
6. Format: section heading → plain explanation → exact punishment."""

                    user_prompt = (
                        f"Chat History:\n{history_text}\n\n"
                        f"Legal Context:\n{context}\n\n"
                        f"Question: {message}\n\n"
                        "Answer strictly from the Legal Context above only."
                    )
                else:
                    system_prompt = "You are an Indian legal assistant for BNS, BNSS and BSA."
                    user_prompt = (
                        f"The user asked: {message}\n\n"
                        "No matching context found. Tell the user you could not find this, "
                        "suggest indiacode.nic.in, and ask them to rephrase with a specific section number."
                    )
            except Exception as e:
                print(f"Retrieval error: {e}")
                system_prompt = "You are an Indian legal assistant."
                user_prompt = f"Retrieval error. User asked: {message}. Apologise and ask to try again."
        else:
            system_prompt = (
                "You are an expert Indian Legal Assistant for BNS, BNSS and BSA. "
                "When a user describes a situation, identify if any Indian law applies and "
                "explain the relevant BNS/BNSS/BSA section, offence, and punishment. "
                "Always try to find a legal angle even if the user does not use legal terms."
            )
            user_prompt = (
                f"Chat History:\n{self.get_history_text()}\n\n"
                f"User: {message}\n\n"
                "Identify if this describes a legal situation under BNS/BNSS/BSA and explain "
                "the relevant law, section, and punishment if applicable."
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
        if is_legal:
            response += "\n\n---\n*⚠️ For educational purposes only. Consult a qualified lawyer for legal advice.*"

        self.remember(message, response)
        return response


# ── FastAPI App ───────────────────────────────────────────────────────────────
assistant = IndianLegalAssistant()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = assistant.generate_response(req.message)
    return JSONResponse({"response": response})

@app.post("/clear")
async def clear():
    assistant.clear_memory()
    return JSONResponse({"status": "cleared"})

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)


# ── ChatGPT-style HTML UI (fully responsive) ──────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0"/>
<title>Indian Legal Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0d0d0d;
    --sidebar: #111111;
    --surface: #1a1a1a;
    --border:  #2a2a2a;
    --accent:  #e8a045;
    --accent2: #c47d2a;
    --text:    #e8e8e8;
    --muted:   #888;
    --radius:  14px;
    --font:    'Sora', sans-serif;
    --mono:    'JetBrains Mono', monospace;
  }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    height: 100dvh;
    display: flex;
    overflow: hidden;
  }

  /* ── Overlay (mobile sidebar backdrop) ── */
  .overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 99;
  }
  .overlay.open { display: block; }

  /* ── Sidebar ── */
  .sidebar {
    width: 260px;
    background: var(--sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 20px 14px;
    gap: 8px;
    flex-shrink: 0;
    transition: transform 0.3s ease;
    z-index: 100;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 8px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
  }

  .logo-icon { font-size: 24px; filter: drop-shadow(0 0 8px var(--accent)); }

  .logo-text {
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1.3;
  }

  .logo-sub { font-size: 10px; color: var(--muted); }

  /* close button — mobile only */
  .sidebar-close {
    display: none;
    position: absolute;
    top: 14px; right: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    width: 32px; height: 32px;
    font-size: 16px;
    cursor: pointer;
    align-items: center;
    justify-content: center;
  }

  .new-chat-btn {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    font-family: var(--font);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s, transform 0.1s;
    margin-bottom: 8px;
  }

  .new-chat-btn:hover  { background: var(--accent2); transform: translateY(-1px); }
  .new-chat-btn:active { transform: translateY(0); }

  .sidebar-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 8px 4px;
  }

  .quick-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .quick-btn:hover {
    background: var(--surface);
    border-color: var(--accent);
    color: var(--accent);
  }

  .law-tags {
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .law-tag {
    font-size: 11px;
    color: var(--muted);
    padding: 4px 8px;
    border-radius: 6px;
    background: var(--surface);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .law-tag span { color: var(--accent); font-weight: 600; }

  /* ── Main ── */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
  }

  .topbar {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    flex-shrink: 0;
    gap: 10px;
  }

  .topbar-left { display: flex; align-items: center; gap: 10px; }

  /* hamburger — mobile only */
  .hamburger {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    width: 36px; height: 36px;
    font-size: 18px;
    cursor: pointer;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .topbar-title { font-size: 15px; font-weight: 600; }

  .topbar-model {
    font-size: 11px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 20px;
    font-family: var(--mono);
    white-space: nowrap;
  }

  /* ── Messages ── */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 0;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
  }

  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .welcome {
    text-align: center;
    padding: 40px 20px;
    max-width: 560px;
    margin: 0 auto;
  }

  .welcome-icon { font-size: 44px; margin-bottom: 14px; filter: drop-shadow(0 0 16px var(--accent)); }
  .welcome h2   { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
  .welcome p    { font-size: 13px; color: var(--muted); line-height: 1.6; }

  .welcome-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
  }

  .chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 7px 13px;
    font-size: 12px;
    color: var(--text);
    cursor: pointer;
    transition: all 0.2s;
  }

  .chip:hover { border-color: var(--accent); color: var(--accent); background: #1e1a14; }

  .message-row {
    display: flex;
    padding: 8px 16px;
    gap: 12px;
    max-width: 860px;
    margin: 0 auto;
    width: 100%;
    animation: fadeUp 0.3s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .avatar.user { background: #2a2a2a; border: 1px solid var(--border); }
  .avatar.bot  { background: #1e1a10; border: 1px solid var(--accent); }

  .bubble {
    flex: 1;
    font-size: 14px;
    line-height: 1.75;
    color: var(--text);
    padding: 4px 0;
    min-width: 0;
    word-break: break-word;
  }

  .bubble.user { color: #d0d0d0; }
  .bubble p    { margin-bottom: 10px; }
  .bubble p:last-child { margin-bottom: 0; }
  .bubble strong { color: var(--accent); }
  .bubble em     { color: #aaa; font-style: italic; }

  .bubble table {
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 12px;
    width: 100%;
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .bubble th, .bubble td {
    border: 1px solid var(--border);
    padding: 7px 10px;
    text-align: left;
    white-space: nowrap;
  }

  .bubble th { background: var(--surface); color: var(--accent); font-weight: 600; }

  .bubble blockquote {
    border-left: 3px solid var(--accent);
    padding: 8px 14px;
    margin: 10px 0;
    background: #1a180e;
    border-radius: 0 8px 8px 0;
    color: #ccc;
    font-size: 13px;
  }

  .bubble code {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
  }

  .bubble hr   { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  .bubble ul, .bubble ol { padding-left: 20px; margin-bottom: 10px; }
  .bubble li   { margin-bottom: 4px; }

  /* ── Typing ── */
  .typing { display: flex; align-items: center; gap: 4px; padding: 8px 0; }

  .typing span {
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
    opacity: 0.7;
  }

  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%           { transform: translateY(-6px); opacity: 1; }
  }

  /* ── Input ── */
  .input-area {
    padding: 12px 16px 16px;
    background: var(--bg);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }

  .input-wrapper {
    max-width: 860px;
    margin: 0 auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    align-items: flex-end;
    gap: 10px;
    padding: 10px 12px;
    transition: border-color 0.2s;
  }

  .input-wrapper:focus-within { border-color: var(--accent); }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.6;
    resize: none;
    max-height: 140px;
    min-height: 24px;
    height: 24px;
    -webkit-appearance: none;
  }

  textarea::placeholder { color: var(--muted); }

  .send-btn {
    background: var(--accent);
    border: none;
    border-radius: 8px;
    width: 36px; height: 36px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.2s, transform 0.1s;
    color: #000;
    font-size: 16px;
  }

  .send-btn:hover    { background: var(--accent2); transform: scale(1.05); }
  .send-btn:disabled { background: var(--border); cursor: not-allowed; transform: none; }

  .input-footer {
    max-width: 860px;
    margin: 6px auto 0;
    font-size: 11px;
    color: var(--muted);
    text-align: center;
  }

  /* ═══════════════════════════════════════════
     MOBILE RESPONSIVE  (≤ 640px)
  ═══════════════════════════════════════════ */
  @media (max-width: 640px) {
    /* Sidebar slides in as drawer */
    .sidebar {
      position: fixed;
      top: 0; left: 0; bottom: 0;
      transform: translateX(-100%);
    }

    .sidebar.open {
      transform: translateX(0);
      box-shadow: 4px 0 24px rgba(0,0,0,0.6);
    }

    .sidebar-close { display: flex; }
    .hamburger     { display: flex; }

    /* Topbar adjustments */
    .topbar { padding: 12px 14px; }
    .topbar-title { font-size: 13px; }
    .topbar-model { font-size: 10px; padding: 3px 8px; }

    /* Messages padding tighter */
    .message-row { padding: 6px 12px; gap: 10px; }
    .avatar      { width: 28px; height: 28px; font-size: 13px; }
    .bubble      { font-size: 13px; }

    /* Welcome screen */
    .welcome         { padding: 30px 16px; }
    .welcome-icon    { font-size: 36px; }
    .welcome h2      { font-size: 17px; }
    .welcome p       { font-size: 12px; }
    .chip            { font-size: 11px; padding: 6px 11px; }

    /* Input area */
    .input-area      { padding: 10px 12px 14px; }
    textarea         { font-size: 16px; } /* prevents iOS zoom */
    .input-footer    { font-size: 10px; }
  }

  /* Tablet (641px – 900px): sidebar still visible but narrower */
  @media (min-width: 641px) and (max-width: 900px) {
    .sidebar { width: 210px; }
    .message-row { padding: 8px 16px; }
  }
</style>
</head>
<body>

<div class="overlay" id="overlay" onclick="closeSidebar()"></div>

<!-- Sidebar -->
<div class="sidebar" id="sidebar">
  <button class="sidebar-close" onclick="closeSidebar()">✕</button>

  <div class="logo">
    <div class="logo-icon">⚖️</div>
    <div>
      <div class="logo-text">Legal Assistant</div>
      <div class="logo-sub">BNS · BNSS · BSA</div>
    </div>
  </div>

  <button class="new-chat-btn" onclick="newChat()">✏️ New Chat</button>

  <div class="sidebar-label">Quick Questions</div>
  <button class="quick-btn" onclick="sq('What is BNS Section 103?')">⚖️ BNS Section 103</button>
  <button class="quick-btn" onclick="sq('What replaced IPC 302?')">🔄 IPC 302 → BNS?</button>
  <button class="quick-btn" onclick="sq('Punishment for theft under BNS')">🔓 Theft punishment</button>
  <button class="quick-btn" onclick="sq('Stalking under which section?')">👁️ Stalking section</button>
  <button class="quick-btn" onclick="sq('Electronic evidence under BSA')">💻 Electronic evidence</button>
  <button class="quick-btn" onclick="sq('What is dowry death under BNS?')">⚠️ Dowry death</button>

  <div class="law-tags">
    <div class="sidebar-label">Active Laws</div>
    <div class="law-tag"><span>BNS</span> Bharatiya Nyaya Sanhita</div>
    <div class="law-tag"><span>BNSS</span> Nagarik Suraksha Sanhita</div>
    <div class="law-tag"><span>BSA</span> Sakshya Adhiniyam</div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="topbar">
    <div class="topbar-left">
      <button class="hamburger" id="hamburger" onclick="openSidebar()">☰</button>
      <div class="topbar-title">Indian Legal Assistant</div>
    </div>
    <div class="topbar-model">llama-3.3-70b · Groq</div>
  </div>

  <div class="messages" id="messages">
    <div class="welcome" id="welcome">
      <div class="welcome-icon">⚖️</div>
      <h2>Indian Legal Assistant</h2>
      <p>Ask about India's new criminal laws — BNS, BNSS and BSA.<br/>
         Describe a situation or ask about a specific section.</p>
      <div class="welcome-chips">
        <div class="chip" onclick="sq('What is BNS Section 103?')">BNS Section 103</div>
        <div class="chip" onclick="sq('What replaced IPC 302?')">What replaced IPC 302?</div>
        <div class="chip" onclick="sq('Stalking under which section?')">Stalking section</div>
        <div class="chip" onclick="sq('Explain electronic evidence under BSA')">Electronic evidence</div>
        <div class="chip" onclick="sq('What is dowry death under BNS?')">Dowry death</div>
        <div class="chip" onclick="sq('Punishment for rape under BNS')">Punishment for rape</div>
      </div>
    </div>
  </div>

  <div class="input-area">
    <div class="input-wrapper">
      <textarea id="input"
        placeholder="Ask about BNS / BNSS / BSA or describe a situation..."
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"
        rows="1"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
    </div>
    <div class="input-footer">
      ⚠️ Educational purposes only · Not a substitute for professional legal advice
    </div>
  </div>
</div>

<script>
  let isLoading = false;

  // ── Sidebar (mobile) ──
  function openSidebar() {
    document.getElementById('sidebar').classList.add('open');
    document.getElementById('overlay').classList.add('open');
  }
  function closeSidebar() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('overlay').classList.remove('open');
  }

  // ── Input helpers ──
  function autoResize(el) {
    el.style.height = '24px';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function hideWelcome() {
    const w = document.getElementById('welcome');
    if (w) w.remove();
  }

  // ── Messages ──
  function addMessage(role, text) {
    hideWelcome();
    const container = document.getElementById('messages');
    const row    = document.createElement('div');
    row.className = 'message-row';

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}`;
    avatar.textContent = role === 'user' ? '👤' : '⚖️';

    const bubble = document.createElement('div');
    bubble.className = `bubble ${role}`;
    bubble.innerHTML  = role === 'bot' ? marked.parse(text) : escapeHtml(text);

    row.appendChild(avatar);
    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  function escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function addTyping() {
    hideWelcome();
    const container = document.getElementById('messages');
    const row = document.createElement('div');
    row.className = 'message-row';
    row.id = 'typing-row';
    row.innerHTML = `
      <div class="avatar bot">⚖️</div>
      <div class="bubble bot">
        <div class="typing"><span></span><span></span><span></span></div>
      </div>`;
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('typing-row');
    if (t) t.remove();
  }

  // ── Send ──
  async function sendMessage() {
    if (isLoading) return;
    const input = document.getElementById('input');
    const msg   = input.value.trim();
    if (!msg) return;

    input.value = '';
    input.style.height = '24px';
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    closeSidebar();

    addMessage('user', msg);
    addTyping();

    try {
      const res  = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      const data = await res.json();
      removeTyping();
      addMessage('bot', data.response);
    } catch(e) {
      removeTyping();
      addMessage('bot', '❌ Connection error. Please try again.');
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
    document.getElementById('messages').innerHTML = `
      <div class="welcome" id="welcome">
        <div class="welcome-icon">⚖️</div>
        <h2>Indian Legal Assistant</h2>
        <p>Ask about India's new criminal laws — BNS, BNSS and BSA.<br/>
           Describe a situation or ask about a specific section.</p>
        <div class="welcome-chips">
          <div class="chip" onclick="sq('What is BNS Section 103?')">BNS Section 103</div>
          <div class="chip" onclick="sq('What replaced IPC 302?')">What replaced IPC 302?</div>
          <div class="chip" onclick="sq('Stalking under which section?')">Stalking section</div>
          <div class="chip" onclick="sq('Explain electronic evidence under BSA')">Electronic evidence</div>
          <div class="chip" onclick="sq('What is dowry death under BNS?')">Dowry death</div>
          <div class="chip" onclick="sq('Punishment for rape under BNS')">Punishment for rape</div>
        </div>
      </div>`;
  }

  document.getElementById('input').focus();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)