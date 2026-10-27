"""Streamlit interface. Verification status and source excerpts are always inspectable."""
import uuid
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from kestrel.graph import stream_question
from kestrel.llm import LLMConfig

load_dotenv(Path(__file__).parent / ".env")
st.set_page_config(page_title="Kestrel Research Assistant", page_icon="🦅")
st.title("🦅 Kestrel Research Assistant")
st.caption("A research prototype over the supplied Kestrel Labs document corpus.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

with st.sidebar:
    configured = LLMConfig.from_env().configured
    st.write("Model-assisted mode" if configured else "Offline source browsing")
    st.caption("Model assessments can be wrong. Inspect the excerpts before relying on an answer.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()


def show_details(message):
    status = message.get("verification_status", "not_requested")
    if status == "performed":
        st.caption(f"Model assessment: {message.get('verdict', '')} · "
                   f"Retrieval attempts: {message.get('retrieval_attempts', 0)}")
    elif status in ("skipped", "failed"):
        st.warning(f"Verification {status}. Sources have not been verified as an answer.")
    if message.get("errors"):
        st.warning("A model stage failed; this response used a fallback.")
    evidence = {e["chunk_id"]: e for e in message.get("evidence", [])}
    if evidence:
        with st.expander("Inspect retrieved sources"):
            for e in evidence.values():
                st.markdown(f"**[{e['chunk_id']}] {e['title']}**")
                st.text(e["text"])
                meta = e.get("metadata", {})
                st.caption(f"Document: {e.get('doc_id', '')} · "
                           f"Version: {meta.get('version', '')} · Published: {meta.get('published', '')}")
    if message.get("citations"):
        st.caption("Cited sources: " + ", ".join(c["chunk_id"] for c in message["citations"]))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_details(message)

if prompt := st.chat_input("Ask about plans, Beacons, KQL, retention, or Warehouse Sync"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        state = {}
        try:
            with st.status("Searching and assessing sources…", expanded=False) as progress:
                for node, _, update in stream_question(
                    prompt, history, conversation_id=st.session_state.conversation_id
                ):
                    state.update(update)
                    st.write(f"{node.capitalize()} completed")
                progress.update(label="Response ready", state="complete")
        except Exception:
            st.error("The workflow could not finish. Check the local index and model configuration.")
        else:
            message = {
                "role": "assistant", "content": state.get("final_answer", ""),
                "citations": state.get("citations", []),
                "verdict": state.get("verification_verdict", ""),
                "verification_status": state.get("verification_status", "not_requested"),
                "retrieval_attempts": state.get("retrieval_attempts", 0),
                "evidence": state.get("evidence", []), "errors": state.get("errors", []),
            }
            st.markdown(message["content"])
            show_details(message)
            st.session_state.messages.append(message)
