"""Streamlit chat UI for the HR Policy AI Agent."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent import HRAgent


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = HRAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    """Render the sidebar with project info and controls."""
    with st.sidebar:
        st.title("🏢 HR Policy Agent")
        st.markdown(
            "An AI assistant that answers questions about company HR policies "
            "using Retrieval-Augmented Generation (RAG)."
        )

        st.divider()

        st.markdown("**What I can help with:**")
        st.markdown(
            "- 🏖️ Vacation & leave policies\n"
            "- 💊 Employee benefits\n"
            "- 📚 Training & development\n"
            "- 📊 Performance evaluations\n"
            "- 📋 Code of conduct"
        )

        st.divider()

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_history()
            st.rerun()

        st.divider()
        st.caption("Built with LangChain, OpenAI, ChromaDB & Streamlit")


def render_sources(sources: list[dict]) -> None:
    """Render source citations in an expandable section."""
    if not sources:
        return

    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        for source in sources:
            name = source.get("source", "Unknown")
            category = source.get("category", "Unknown")
            page = source.get("page")
            line = f"- **{name}** — {category}"
            if page is not None:
                line += f" (Page {page + 1})"
            st.markdown(line)


def main() -> None:
    st.set_page_config(
        page_title="HR Policy Assistant",
        page_icon="🏢",
        layout="centered",
    )

    init_session_state()
    render_sidebar()

    st.header("💬 HR Policy Assistant")
    st.caption("Ask me anything about company HR policies. I'll cite my sources.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Ask about HR policies..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching HR policies..."):
                result = st.session_state.agent.process(prompt)

            st.markdown(result["answer"])
            render_sources(result["sources"])

        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "query_type": result["query_type"],
        })


if __name__ == "__main__":
    main()
