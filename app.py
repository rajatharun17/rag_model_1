import streamlit as st
import os
import json
import time
from dotenv import load_dotenv

from src.chunking import AdvancedChunker
from src.vector_store import FastVectorStore
from src.harness import OrchestrationHarness
from src.stt_engine import ElevenLabsSTTEngine  # Or SarvamSTTEngine

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Voice-Enabled RAG System",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Voice-Enabled RAG Pipeline")
st.caption("Speak a question → Transcribe → Vector Search → Grounded Answer (<200ms target)")

# Cache vector database loading so it doesn't re-index on every UI refresh
@st.cache_resource(show_spinner=True)
def initialize_rag_system():
    DATASET_PATH = "data/tamil/tamil_rag_10000.jsonl"
    raw_records = []
    
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Load 1,000 documents into memory
                    break
                raw_records.append(json.loads(line))
    else:
        # Fallback dummy dataset
        raw_records = [
            {"id": "1", "title": "Tamil Nadu", "passage": "Chennai is the capital of Tamil Nadu."},
            {"id": "2", "title": "RAG Models", "passage": "Retrieval-Augmented Generation combines vector retrieval with language modeling."}
        ]

    # Apply hybrid chunking
    all_chunks = []
    for rec in raw_records:
        all_chunks.extend(AdvancedChunker.process_record(rec, strategy="hybrid"))

    # Index into in-memory Qdrant
    vector_store = FastVectorStore()
    vector_store.index_chunks(all_chunks)
    
    harness = OrchestrationHarness(vector_store)
    stt_engine = ElevenLabsSTTEngine()  # Swap with SarvamSTTEngine() if using Sarvam
    
    return harness, stt_engine

# Initialize
harness, stt_engine = initialize_rag_system()

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
language_code = st.sidebar.selectbox("STT Language Code", ["ta", "en", "hi"], index=0)
temp_audio_path = "data/temp_recorded_audio.wav"

# Streamlit Native Browser Audio Recorder Input
st.subheader("1. Record Your Voice Question")
audio_value = st.audio_input("Click the microphone icon to record your question:")

if audio_value:
    # Save recorded audio file locally
    os.makedirs("data", exist_ok=True)
    with open(temp_audio_path, "wb") as f:
        f.write(audio_value.read())

    st.audio(temp_audio_path, format="audio/wav")

    if st.button("🚀 Process Voice Query", type="primary"):
        st.divider()
        
        # Step A: Speech to Text
        st.subheader("2. Speech-to-Text Transcription")
        with st.spinner("Transcribing audio..."):
            stt_start = time.perf_counter()
            stt_res = stt_engine.transcribe(temp_audio_path, language_code=language_code)
            stt_latency = (time.perf_counter() - stt_start) * 1000

        if not stt_res["success"]:
            st.error(f"STT Error: {stt_res.get('error', 'Transcription failed')}")
        else:
            query_text = stt_res["transcript"]
            st.success(f"**Transcript:** {query_text}")
            st.caption(f"STT Latency: {stt_latency:.2f} ms")

            # Step B: Orchestration Harness & Vector Retrieval
            # Step B: Orchestration Harness & Vector Retrieval
            st.subheader("3. RAG Pipeline Output")
            with st.spinner("Searching vector database..."):
                pipeline_result = harness.run_pipeline(query_text)

            # Safely extract values with defaults
            status = pipeline_result.get("status", "ERROR")
            answer = pipeline_result.get("answer", "No answer generated.")
            reason = pipeline_result.get("reason", "No specific reason provided.")

            if status == "SUCCESS":
                st.success("✅ **Grounded Answer Generated**")
                st.markdown(f"### Answer:\n{answer}")
    
                with st.expander("🔍 View Retrieved Context"):
                    st.text(pipeline_result.get("context", ""))

                col1, col2, col3 = st.columns(3)
                col1.metric("Vector Retrieval Latency", f"{pipeline_result.get('retrieval_ms', 0):.2f} ms")
                col2.metric("Total RAG Latency", f"{pipeline_result.get('total_latency_ms', 0):.2f} ms")
                col3.metric("Pipeline Status", status)

            elif status == "UNGROUNDED":    
                st.warning("⚠️ **Low Groundedness Score**")
                st.write(answer)
                st.caption(f"Reason: {reason}")

            else:  # REJECTED or ERROR
                st.error("❌ **Pipeline Safety Gate Triggered**")
                st.write(f"**Status:** {status}")
                st.write(f"**Reason:** {reason}")
                if "answer" in pipeline_result:
                    st.write(f"**Message:** {answer}")