import streamlit as st
import asyncio
from src.core.langraph_workflow import ingest_from_medline, query_pipeline
from src.core.vector_store import upsert_embeddings



st.set_page_config(page_title="MedlinePlus Chat", page_icon="🩺", layout="wide")
st.title("MedlinePlus Medical Assistant")
st.markdown("Ask questions and get answers.")

# Initialize session state to store chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Current session/chat ID
thread_id = "default_conversation"

# Displaying chat history from session state
for message_obj in st.session_state.messages:
    with st.chat_message(message_obj["role"]):
        st.markdown(message_obj["content"])


if prompt := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking about the answer..."):
            try:
                llm_response = asyncio.run(query_pipeline(prompt, thread_id=thread_id))
            except Exception as e:
                llm_response = f"Error receiving response: {e}"

        st.markdown(llm_response)

    st.session_state.messages.append({"role": "assistant", "content": llm_response})


col1, col2 = st.columns(2)

with col1:
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

with col2:
    # Place the input outside the button callback so value persists across reruns
    search_term = st.text_input("Enter a term to search for:")
    if st.button("Download medical data"):
        if not search_term:
            st.warning("Please enter a term before downloading data.")
        else:
            with st.spinner("Loading data from MedlinePlus..."):
                try:
                    docs = ingest_from_medline(search_term)

                    if not docs:
                        st.info(f"No documents found for '{search_term}'.")
                    else:
                        texts = [doc.page_content for doc in docs]
                        metadata_list = [{"title": doc.metadata.get("title", ""),
                                          "summary": doc.page_content} for doc in docs]

                        upsert_embeddings(texts, metadata_list)

                        st.success(f"Documents downloaded and saved: {len(docs)} for the term '{search_term}'")
                except Exception as e:
                    st.error(f"Error loading data: {e}")














