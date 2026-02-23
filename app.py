import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

from utils.preprocessing import preprocess_pdf
from utils.vector_store import ChromaVectorStore
from utils.context_manager import build_prompt

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="📖",
    layout="wide",
)

st.markdown("""
<style>
.stTextInput > div > div > input { border-radius: 20px !important; padding: 10px !important; }
.doc-card { background: #f8f9fa; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; }
.citation { font-size: 0.78em; color: #6c757d; font-style: italic; }
</style>
""", unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = ChromaVectorStore()

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs: dict[str, str] = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Add it to your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

vs: ChromaVectorStore = st.session_state.vector_store
with st.sidebar:
    st.header("📂 Document Library")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs. They will be indexed automatically.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.indexed_docs:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        chunks  = preprocess_pdf(uf, doc_name=uf.name)
                        doc_id  = vs.index_document(chunks, doc_name=uf.name)
                        st.session_state.indexed_docs[uf.name] = doc_id
                        st.success(f"✅ {uf.name}")
                    except Exception as e:
                        st.error(f"Error indexing {uf.name}: {e}")

    st.divider()

    if st.session_state.indexed_docs:
        st.subheader(f"📄 Indexed ({len(st.session_state.indexed_docs)})")

        for doc_name, doc_id in list(st.session_state.indexed_docs.items()):
            with st.container():
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{doc_name}**")
                if col2.button("🗑️", key=f"del_{doc_id}", help="Remove document"):
                    vs.delete_document(doc_id)
                    del st.session_state.indexed_docs[doc_name]
                    st.rerun()

                # Per-document summary button
                if st.button(f"📝 Summarize", key=f"sum_{doc_id}"):
                    with st.spinner("Generating summary…"):
                        try:
                            hits = vs.retrieve(
                                "Provide a comprehensive overview of this document.",
                                top_k=6,
                                doc_ids=[doc_id],
                            )
                            summary_prompt = (
                                "You are a document summarizer. Based on the following excerpts, "
                                "write a clear, structured summary (key topics, main findings, "
                                "important details). Cite page numbers where relevant.\n\n"
                                + "\n\n".join(
                                    f"[Page {h['page']}]\n{h['content']}" for h in hits
                                )
                            )
                            summary = model.generate_content(summary_prompt).text
                            st.info(summary)
                        except Exception as e:
                            st.error(f"Summary error: {e}")
    else:
        st.info("No documents indexed yet. Upload PDFs above.")

    st.divider()

    if st.button("🧹 Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        export_lines = []
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            export_lines.append(f"### {role}\n{msg['content']}\n")
            if msg.get("sources"):
                src_lines = [
                    f"  - {s['doc_name']} (Page {s['page']}, relevance: {1 - s['distance']:.2f})"
                    for s in msg["sources"]
                ]
                export_lines.append("**Sources:**\n" + "\n".join(src_lines) + "\n")
            export_lines.append("---\n")

        export_text = "\n".join(export_lines)

        st.download_button(
            label="💾 Export Chat",
            data=export_text,
            file_name="chat_export.txt",
            mime="text/markdown",
            use_container_width=True,
        )

    st.caption(
        f"Total chunks indexed: **{vs.total_chunks()}**"
    )

st.title("📖 GenAI Multi-Document Assistant")

if not st.session_state.indexed_docs:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
else:
    docs_list = ", ".join(st.session_state.indexed_docs.keys())
    st.caption(f"Querying across: {docs_list}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['doc_name']}** — Page {src['page']} "
                        f"*(relevance score: {1 - src['distance']:.2f})*"
                    )

question = st.chat_input("Ask anything about your documents…")

if question:
    if not st.session_state.indexed_docs:
        st.warning("⚠️ Please upload at least one PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    hits = vs.retrieve(question, top_k=5)

                    if not hits:
                        answer = "I couldn't find any relevant information in the indexed documents."
                        sources = []
                    else:
                        prompt = build_prompt(
                            question=question,
                            hits=hits,
                            chat_history=st.session_state.chat_history[:-1],  # exclude current question
                        )
                        response = model.generate_content(prompt)
                        answer   = response.text
                        sources  = [
                            {"doc_name": h["doc_name"], "page": h["page"], "distance": h["distance"]}
                            for h in hits
                        ]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"- **{src['doc_name']}** — Page {src['page']} "
                                    f"*(relevance: {1 - src['distance']:.2f})*"
                                )

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")
