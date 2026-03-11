"""
Indian Legal Assistant — BNS / BNSS / BSA Edition
Powered by: Groq API + LLaMA 3.3 (llama-3.3-70b-versatile)

Run i.py first to build the vector database.

Install dependencies:
    uv pip install langchain langchain-community langchain-core \
                   langchain-huggingface langchain-groq \
                   langchain-chroma langchain-text-splitters \
                   chromadb sentence-transformers \
                   gradio python-dotenv pypdf

.env file:
    groq_key=gsk_xxxxxxxxxxxx
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ── LangChain 1.x compatible imports ─────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma                           # updated package
from langchain_groq import ChatGroq
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("groq_key", "").strip().strip("'\"")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # current active model on Groq

# ── IPC → BNS mapping ─────────────────────────────────────────────────────────
IPC_TO_BNS = {
    "302":  ("103", "BNS", "Murder"),
    "304":  ("105", "BNS", "Culpable homicide not amounting to murder"),
    "304B": ("80",  "BNS", "Dowry death"),
    "306":  ("108", "BNS", "Abetment of suicide"),
    "307":  ("109", "BNS", "Attempt to murder"),
    "323":  ("115", "BNS", "Voluntarily causing hurt"),
    "354":  ("74",  "BNS", "Assault on woman to outrage modesty"),
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

        # ── Embeddings ────────────────────────────────────────────────────────
        print("\nLoading embedding model ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},   # GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✓ Embeddings ready")

        # ── Vector DB ─────────────────────────────────────────────────────────
        print("Loading vector database ...")
        try:
            self.db = Chroma(
                persist_directory="vector_db",
                embedding_function=self.embeddings,
            )
            print("✓ Vector database loaded")
        except Exception as e:
            print(f"✗ Vector DB error: {e}")
            print("  Run i.py first to build the database!")
            self.db = None

        self.retriever = (
            self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            if self.db else None
        )

        # ── Groq LLM ─────────────────────────────────────────────────────────
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
            print("  Check your groq_key in .env")
            self.llm = None

        # ── Chat history (simple list, no LangChain memory needed) ───────────
        self.chat_history = []

        # ── Legal keywords ────────────────────────────────────────────────────
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
        ]

        print("\n✓ Legal Assistant is ready!\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_legal_query(self, message: str) -> bool:
        msg = message.lower()
        if re.search(r'(section|sec|bns|bnss|bsa|ipc)[.\s]*(\d+[A-Z]?)', msg):
            return True
        return any(kw in msg for kw in self.legal_keywords)

    def check_ipc_reference(self, message: str) -> str:
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

    def get_history_text(self) -> str:
        if not self.chat_history:
            return "None"
        lines = []
        for m in self.chat_history[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def remember(self, user_msg: str, ai_msg: str):
        self.chat_history.append({"role": "user",      "content": user_msg})
        self.chat_history.append({"role": "assistant", "content": ai_msg})

    def clear_memory(self):
        self.chat_history.clear()

    # ── Main response generator ───────────────────────────────────────────────

    def generate_response(self, message: str, history) -> str:
        if not self.llm:
            return "❌ Groq LLM not connected. Check your groq_key in .env"

        is_legal = self.is_legal_query(message)
        ipc_note = self.check_ipc_reference(message)

        # ── Greeting ──────────────────────────────────────────────────────────
        greetings = ["hello", "hi", "hey", "greetings", "good morning",
                     "good afternoon", "good evening", "namaste"]
        if any(g in message.lower() for g in greetings):
            resp = (
                "Namaste! 🙏 I'm your **Indian Legal Assistant**, powered by LLaMA 3.3 (70B).\n\n"
                "I cover India's **new criminal laws** (effective 1st July 2024):\n\n"
                "| New Law | Replaces |\n|---|---|\n"
                "| 📘 **BNS** – Bharatiya Nyaya Sanhita, 2023 | IPC, 1860 |\n"
                "| 📗 **BNSS** – Bharatiya Nagarik Suraksha Sanhita, 2023 | CrPC, 1973 |\n"
                "| 📙 **BSA** – Bharatiya Sakshya Adhiniyam, 2023 | Evidence Act, 1872 |\n\n"
                "Ask about any section, offence, or punishment. "
                "You can also ask about **old IPC numbers** — I'll map them to the new law!"
            )
            self.remember(message, resp)
            return resp

        # ── Legal query with RAG ──────────────────────────────────────────────
        if is_legal and self.retriever:
            try:
                docs         = self.retriever.invoke(message)
                history_text = self.get_history_text()

                if docs:
                    parts = []
                    for i, doc in enumerate(docs[:4], 1):
                        law = doc.metadata.get("law", "Indian Law")
                        sec = doc.metadata.get("section", "?")
                        parts.append(
                            f"[Source {i} — {law}, Section {sec}]\n{doc.page_content[:600]}"
                        )
                    context = "\n\n".join(parts)

                    system_prompt = """You are an expert Indian criminal law assistant covering:
- Bharatiya Nyaya Sanhita (BNS), 2023 — replaces IPC 1860
- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 — replaces CrPC 1973
- Bharatiya Sakshya Adhiniyam (BSA), 2023 — replaces Evidence Act 1872

STRICT RULES:
1. Answer ONLY from the Legal Context provided below.
2. If the answer is NOT in the context, say: "I could not find this in my database. Please check indiacode.nic.in or consult a lawyer."
3. NEVER guess or hallucinate any legal provision or punishment.
4. Always cite the exact law (BNS/BNSS/BSA) and section number.
5. Quote punishments verbatim from the law text.
6. Format answer clearly: section heading → plain explanation → exact punishment."""

                    user_prompt = (
                        f"Chat History:\n{history_text}\n\n"
                        f"Legal Context:\n{context}\n\n"
                        f"Question: {message}\n\n"
                        "Answer strictly from the Legal Context above only. Do not use outside knowledge."
                    )

                else:
                    system_prompt = "You are an Indian legal assistant for BNS, BNSS and BSA."
                    user_prompt   = (
                        f"The user asked: {message}\n\n"
                        "No matching context found in the database. Tell the user you could not "
                        "find this section, suggest indiacode.nic.in, and ask them to rephrase "
                        "with a specific section number."
                    )

            except Exception as e:
                print(f"Retrieval error: {e}")
                system_prompt = "You are an Indian legal assistant."
                user_prompt   = f"Retrieval error. User asked: {message}. Apologise and ask to try again."

        else:
            system_prompt = (
                "You are a helpful Indian Legal Assistant for BNS, BNSS and BSA. "
                "Keep responses brief and friendly. Redirect non-legal topics back to Indian law."
            )
            user_prompt = f"Chat History:\n{self.get_history_text()}\n\nUser: {message}"

        # ── Call Groq ─────────────────────────────────────────────────────────
        try:
            result   = self.llm.invoke([
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


# ── Gradio UI ─────────────────────────────────────────────────────────────────
assistant = IndianLegalAssistant()

with gr.Blocks(theme=gr.themes.Soft(), title="Indian Legal Assistant") as demo:
    gr.Markdown("""
    # ⚖️ Indian Legal Assistant — New Laws Edition
    *Powered by **LLaMA 3.3 (70B)** via Groq*

    | New Law | Replaces |
    |---|---|
    | 📘 **BNS** – Bharatiya Nyaya Sanhita, 2023 | IPC, 1860 |
    | 📗 **BNSS** – Bharatiya Nagarik Suraksha Sanhita, 2023 | CrPC, 1973 |
    | 📙 **BSA** – Bharatiya Sakshya Adhiniyam, 2023 | Evidence Act, 1872 |

    You can also ask about **old IPC section numbers** — I'll give you the new equivalent!
    """)

    chatbot = gr.Chatbot(height=450)

    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What is BNS Section 103?  /  What replaced IPC 302?",
            scale=4, lines=2,
        )
        submit = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        clear_chat   = gr.Button("Clear Chat")
        clear_memory = gr.Button("Clear Memory", variant="secondary")

    gr.Examples(
        examples=[
            "What is BNS Section 103?",
            "What replaced IPC Section 302?",
            "Explain punishment for theft under BNS",
            "What is BNS Section 64?",
            "What is BNSS Section 187?",
            "Explain electronic evidence admissibility under BSA",
            "What is dowry death under BNS?",
            "What is the difference between BNS and old IPC?",
        ],
        inputs=msg,
    )

    gr.Markdown("---\n*⚠️ General information only. Not a substitute for professional legal advice.*")

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        response     = assistant.generate_response(message, chat_history)
        chat_history = (chat_history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        return "", chat_history

    def clear_all():
        assistant.clear_memory()
        return []

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_chat.click(lambda: [], None, chatbot, queue=False)
    clear_memory.click(clear_all, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)