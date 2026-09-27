"""Streamlit is a thin API client; it never opens the vector database."""
import os
import httpx
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "180")) * 2 + 60
st.set_page_config(page_title="Evidence Desk", page_icon="📚", layout="wide")
st.title("Evidence Desk")
st.caption("Ask questions of your documents. Inspect the evidence behind every answer.")
st.info("Historical NIST guidance, not a compliance determination. Each question is independent; repeat names in follow-ups.")

with st.sidebar:
    st.header("Documents")
    file = st.file_uploader("Upload a text PDF", type=["pdf"])
    if file and st.button("Index document"):
        with st.spinner("Extracting and indexing…"):
            try:
                r = httpx.post(API + "/documents", files={"file": (file.name, file.getvalue(), "application/pdf")}, timeout=TIMEOUT)
                r.raise_for_status(); st.success(f"Indexed {r.json()['chunks']} chunks")
            except httpx.HTTPError as e:
                st.error(f"Upload failed: {e}")
    try:
        r = httpx.get(API + "/documents", timeout=10); r.raise_for_status()
        docs = r.json()
    except httpx.HTTPError:
        docs = []; st.warning("Start the API or wait for its embedding model to load.")
    by_id = {d["document_id"]: d["title"] for d in docs}
    selected = st.multiselect("Search scope (empty = all)", list(by_id), format_func=by_id.get)
    st.caption("PDF bytes and models remain local in the default Ollama profile.")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

def render(result):
    if result["mode"] == "extractive":
        st.warning("Evidence-only mode: these are retrieved passages, not a generated answer.")
    if result["abstain"]:
        st.write("I cannot answer from the available evidence. " + result["reason"])
    for claim in result["claims"]:
        refs = " ".join(f"[{e['source_id']}]" for e in claim["evidence"])
        st.write(claim["text"] + " " + refs)
    for sid, source in result["sources"].items():
        with st.expander(f"[{sid}] {source['title']} · PDF page {source['page']}"):
            if source["section"]:
                st.caption("Detected heading (heuristic): " + source["section"])
            st.text(source["text"])
            # Download bytes server-side: works when the API hostname is internal Docker DNS.
            try:
                pdf = httpx.get(API + f"/documents/{source['document_id']}/pdf", timeout=20)
                pdf.raise_for_status()
                st.download_button("Download source PDF", pdf.content, file_name="source.pdf",
                                   mime="application/pdf", key=f"{id(result)}-{sid}")
            except httpx.HTTPError:
                st.caption("Source PDF unavailable")
    st.caption(f"Total: {result['timing_ms']['total']/1000:.2f}s · citations verified for provenance, not semantic truth")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render(message["result"]) if "result" in message else st.write(message["text"])
if question := st.chat_input("What does the report say about…?"):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and checking citations…"):
            try:
                r = httpx.post(API + "/query", json={"question": question, "document_ids": selected}, timeout=TIMEOUT)
                r.raise_for_status(); result = r.json()
                render(result)
                st.session_state.messages.append({"role": "assistant", "result": result})
            except httpx.HTTPError as e:
                st.error(f"Query failed: {e}. Check the API and Ollama logs.")
