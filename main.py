import streamlit as st
import asyncio
from src.core.langraph_workflow import ingest_from_medline, query_pipeline
from src.core.vector_store import upsert_embeddings

st.set_page_config(page_title="MedlinePlus Chat", page_icon="🩺", layout="wide")
st.title("MedlinePlus Medical Assistant")
st.markdown("First, enter the topic, upload the data, and then ask questions.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "search_term" not in st.session_state:
    st.session_state.search_term = ""

# Current session/chat ID
thread_id = "default_conversation"


st.subheader("Downloading data from MedlinePlus")
st.session_state.search_term = st.text_input("Enter a search topic:", st.session_state.search_term)

if st.button("Download data"):
    if not st.session_state.search_term:
        st.warning("Please enter a subject.")
    else:
        with st.spinner("Loading data from MedlinePlus..."):
            try:
                docs = ingest_from_medline(st.session_state.search_term)

                if not docs:
                    st.info(f"Data on the topic '{st.session_state.search_term}' not found.")
                else:
                    texts = [doc.page_content for doc in docs]
                    metadata_list = [{"title": doc.metadata.get("title", ""),
                                      "summary": doc.page_content} for doc in docs]

                    upsert_embeddings(texts, metadata_list)
                    st.success(f"Data downloaded ({len(docs)} documents) for the topic '{st.session_state.search_term}'")
                    st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error during download: {e}")


if st.session_state.data_loaded:
    st.subheader("Chat on the topic")


    for message_obj in st.session_state.messages:
        with st.chat_message(message_obj["role"]):
            st.markdown(message_obj["content"])

    if prompt := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Forming a response..."):
                try:
                    llm_response = asyncio.run(query_pipeline(prompt, thread_id=thread_id))
                except Exception as e:
                    llm_response = f"Error: {e}"

            st.markdown(llm_response)

        st.session_state.messages.append({"role": "assistant", "content": llm_response})


    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("To ask a question, first download the data on the topic.")
