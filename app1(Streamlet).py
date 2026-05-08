import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import PyPDF2

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI – Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  #MainMenu, footer, header { visibility: hidden; }

  .stApp { background-color: #0f1117; color: #e8e8e0; }

  section[data-testid="stSidebar"] {
    background-color: #16181f;
    border-right: 1px solid #2a2d38;
  }
  section[data-testid="stSidebar"] * { color: #c5c5bb !important; }

  .lex-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0ede4;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.1;
  }
  .lex-subtitle { font-size: 0.9rem; color: #6b6b60; margin-top: 4px; }

  .lex-badge {
    display: inline-block;
    background: #1e2a1e;
    color: #4caf6e;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #2d4a30;
    margin-bottom: 10px;
  }

  .upload-zone {
    background: #16181f;
    border: 1.5px dashed #2a2d38;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
  }

  .latest-card {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1.2rem;
  }

  .latest-question {
    font-size: 0.82rem;
    color: #4a7fa0;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  .latest-q-text {
    background: #1c2b3a;
    border: 1px solid #1e3a50;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #d0e8f5;
    font-size: 0.93rem;
    margin-bottom: 12px;
  }

  .latest-answer {
    font-size: 0.82rem;
    color: #4caf6e;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .latest-a-text {
    background: #1a1c24;
    border: 1px solid #252830;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    color: #e0ddd4;
    font-size: 0.93rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  .old-user {
    background: #1c2b3a;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    color: #d0e8f5;
    font-size: 0.88rem;
    margin-bottom: 6px;
    text-align: right;
  }

  .old-bot {
    background: #1a1c24;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    color: #c5c5bb;
    font-size: 0.88rem;
    margin-bottom: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .old-label-u {
    font-size: 0.7rem; color: #4a7fa0;
    font-weight: 600; letter-spacing: 0.7px;
    text-transform: uppercase; text-align: right;
    margin-bottom: 2px;
  }

  .old-label-b {
    font-size: 0.7rem; color: #4caf6e;
    font-weight: 600; letter-spacing: 0.7px;
    text-transform: uppercase;
    margin-bottom: 2px;
  }

  .stTextInput > div > div > input {
    background-color: #16181f !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 10px !important;
    color: #e8e8e0 !important;
    padding: 0.65rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #4caf6e !important;
    box-shadow: 0 0 0 2px rgba(76,175,110,0.15) !important;
  }

  .stButton > button {
    background: #1e3a28 !important;
    color: #4caf6e !important;
    border: 1px solid #2d5c3a !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    background: #254a32 !important;
    border-color: #4caf6e !important;
  }

  hr { border-color: #1e2028; }

  .pdf-card {
    background: #16181f;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
  }
  .pdf-name { font-weight: 600; font-size: 0.88rem; color: #e0ddd4; }
  .pdf-stat { font-size: 0.78rem; color: #6b6b60; margin-top: 2px; }
  .status-dot {
    width: 8px; height: 8px;
    background: #4caf6e; border-radius: 50%;
    display: inline-block; margin-right: 5px;
  }

  .empty-state {
    text-align: center; padding: 3rem 1rem; color: #3a3d48;
  }
  .empty-state-icon { font-size: 3rem; margin-bottom: 1rem; }
  .empty-state-text { font-size: 0.9rem; line-height: 1.7; }

  .stSpinner > div { border-top-color: #4caf6e !important; }
</style>
""", unsafe_allow_html=True)

# ── Load env + Groq client ────────────────────────────────────────────────────
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("pdf_text", ""),
    ("pdf_name", ""),
    ("pdf_words", 0),
    ("prefill", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=1200):
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for word in words:
        cur_len += len(word) + 1
        if cur_len > chunk_size:
            chunks.append(" ".join(cur))
            cur, cur_len = [word], len(word)
        else:
            cur.append(word)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def get_relevant_chunks(chunks, query, max_chunks=2):
    q_words = set(query.lower().split())
    scored = sorted(chunks, key=lambda c: len(q_words & set(c.lower().split())), reverse=True)
    return " ".join(scored[:max_chunks])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="lex-badge">⚖ LexAI</div>', unsafe_allow_html=True)
    st.markdown("### Session History")
    st.markdown("---")

    if not st.session_state.history:
        st.markdown('<div style="font-size:0.82rem;color:#3a3d48;margin-top:0.5rem;">No queries yet.</div>', unsafe_allow_html=True)
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            q = entry["query"]
            short = (q[:48] + "…") if len(q) > 48 else q
            st.markdown(f"""
            <div style="background:#1a1c24;border:1px solid #2a2d38;border-radius:10px;
              padding:0.55rem 0.8rem;margin-bottom:0.45rem;font-size:0.8rem;color:#9a9a90;">
              <span style="color:#4caf6e;font-weight:600;font-size:0.7rem;">
                Q{len(st.session_state.history)-i}
              </span>&nbsp;{short}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear session"):
        for k in ["history", "pdf_text", "pdf_name", "pdf_words", "prefill"]:
            st.session_state[k] = [] if k == "history" else ""
        st.rerun()

    st.markdown(
        '<div style="font-size:0.72rem;color:#3a3d48;margin-top:2rem;">'
        'Powered by Groq · llama-3.1-8b-instant</div>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="lex-title">⚖️ LexAI</p>', unsafe_allow_html=True)
st.markdown('<p class="lex-subtitle">AI-powered legal document analysis & Q&A</p>', unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 1.8], gap="large")

# ── Left: Upload + Quick prompts ──────────────────────────────────────────────
with left:
    st.markdown("#### 📂 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Extracting text…"):
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join(p.extract_text() or "" for p in reader.pages)
                st.session_state.pdf_text = text
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_words = len(text.split())
            st.success("Document ready!")

        chunks_count = len(chunk_text(st.session_state.pdf_text))
        st.markdown(f"""
        <div class="pdf-card">
          <div style="font-size:1.5rem;margin-bottom:6px;">📄</div>
          <div class="pdf-name">{st.session_state.pdf_name}</div>
          <div class="pdf-stat">
            <span class="status-dot"></span>
            {st.session_state.pdf_words:,} words &nbsp;·&nbsp; {chunks_count} chunks
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
          <div style="font-size:2rem;margin-bottom:0.6rem;">📁</div>
          <div style="font-size:0.88rem;color:#4a4d58;">Drop a PDF legal document here</div>
          <div style="font-size:0.78rem;color:#3a3d48;margin-top:4px;">
            Contracts · Agreements · Briefs · Filings
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.pdf_text:
        st.markdown("#### 💡 Quick prompts")
        suggestions = [
            "Summarize this document",
            "What are the key obligations?",
            "Identify any risks or red flags",
            "What are the termination clauses?",
            "List all parties involved",
            "What are the payment terms?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s}"):
                st.session_state.prefill = s
                st.rerun()

# ── Right: Chat ───────────────────────────────────────────────────────────────
with right:
    st.markdown("#### 💬 Chat")

    # Input always visible at top
    prefill_val = st.session_state.prefill
    user_input = st.text_input(
        "Ask a question…",
        value=prefill_val,
        placeholder="e.g. What are the indemnification clauses?",
        label_visibility="collapsed",
        key="chat_input",
    )
    st.session_state.prefill = ""

    send_col, clear_col = st.columns([1, 1])
    with send_col:
        send = st.button("Send →", use_container_width=True)
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Handle send ───────────────────────────────────────────────────────────
    if send and user_input.strip():
        with st.spinner("Analysing document…"):
            try:
                if st.session_state.pdf_text:
                    chunks = chunk_text(st.session_state.pdf_text)
                    context = get_relevant_chunks(chunks, user_input)
                    prompt = f"Legal Document (relevant excerpt):\n{context}\n\nQuestion: {user_input}"
                else:
                    prompt = user_input

                resp = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are LexAI, a professional legal assistant. "
                                "Answer clearly and precisely based on the document provided. "
                                "Use plain English. Highlight important clauses, obligations, or risks. "
                                "If the information is not in the document, say so honestly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=MODEL,
                    max_tokens=1024,
                )
                response = resp.choices[0].message.content
            except Exception as e:
                response = f"⚠️ Error: {e}"

        st.session_state.history.append({"query": user_input, "response": response})
        st.rerun()  # rerun brings latest answer to top automatically

    # ── Display ───────────────────────────────────────────────────────────────
    if st.session_state.history:
        latest = st.session_state.history[-1]

        # Latest answer always shown prominently at top
        st.markdown(f"""
        <div class="latest-card">
          <div class="latest-question">You</div>
          <div class="latest-q-text">{latest['query']}</div>
          <div class="latest-answer"><span class="status-dot"></span>LexAI</div>
          <div class="latest-a-text">{latest['response']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Older messages tucked into expander
        if len(st.session_state.history) > 1:
            with st.expander(f"📜 {len(st.session_state.history) - 1} earlier message(s)"):
                for entry in reversed(st.session_state.history[:-1]):
                    st.markdown(
                        f'<div class="old-label-u">You</div>'
                        f'<div class="old-user">{entry["query"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="old-label-b">LexAI</div>'
                        f'<div class="old-bot">{entry["response"]}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">⚖️</div>
          <div class="empty-state-text">
            Upload a legal document on the left,<br>
            then ask any question about it here.
          </div>
        </div>
        """, unsafe_allow_html=True)
