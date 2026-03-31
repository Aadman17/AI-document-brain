# Project Report: AI Document Brain

**Date:** 2026-03-31

## Abstract

AI Document Brain is a lightweight, user-friendly Streamlit application that converts uploaded PDFs and text files into a searchable document brain. It uses sentence-transformers embeddings and FAISS for vector search, Hugging Face summarization for extractive/abstractive summaries, and a SQuAD-style QA pipeline for targeted question answering. This report documents the design, key improvements, features, evaluation and future directions.

## 1. Introduction

Organizations and researchers often have many documents and need a simple way to search, summarize and extract answers. This project packages modern embedding and transformer techniques into an interactive UI for non-technical users.

## 2. Objectives

- Provide semantic search across multiple documents
- Enable quick document summaries
- Allow question-answering over the uploaded documents
- Be easy to run locally using Streamlit

## 3. Design & Architecture

- Front-end: Streamlit UI with sidebar for uploads and tabs for functionality.
- Embedding & Search: sentence-transformers (all-MiniLM-L6-v2) for embeddings, indexed with FAISS (IndexFlatL2).
- Summarization & QA: Hugging Face pipelines: BART-large-cnn for summarization, RoBERTa SQuAD2 model for QA.
- Persistence: In-memory session state (Streamlit) with @st.cache_resource used to store loaded models.

Diagram (high-level):

```
[User Browser] <-> [Streamlit App]
                     |-- File parsing (PyPDF2)
                     |-- Chunking (word-based)
                     |-- Embedding (SentenceTransformer)
                     |-- Indexing (FAISS)
                     |-- Summarization (Hugging Face)
                     |-- QA (Hugging Face)
```

## 4. Key Improvements

- Suppressed noisy warnings by configuring logging and controlling verbose outputs
- Added robust try/except blocks and user-facing error messages
- Progress bars and status text during long-running tasks
- UI improvements: metrics, clear indexing status, tabs and expandable contexts
- Models cached and loaded once to save memory and reduce startup time
- Clear All Documents button to reset app state

## 5. New Features

- Display of similarity scores (converted from distance to similarity via a simple transform in UI)
- Document statistics (word count, character count, chunk count) displayed per document
- Expandable context view in Q&A to inspect chunks used to form answers
- Confidence shown for QA answers
- Progress bar during indexing
- Tips section in sidebar for better UX

## 6. Implementation Details

- Chunking: Word-based chunks (default 500 words). This balances context window length and embedding costs.
- Embeddings: embedder.encode(texts, convert_to_numpy=True) is used to get a NumPy array suitable for FAISS.
- Indexing: FAISS IndexFlatL2 is used for simplicity and speed on CPU — easy to swap for IVF/HNSW on larger corpora.
- Summarization: Input is truncated to ~1024 words for reliability; min_length and max_length are set for sensible outputs.
- QA: Context is truncated to ~512 words for the QA model to avoid token limits.

## 7. Evaluation

- Functional testing: Manual tests with academic papers, blog posts and short reports ensured chunking, indexing, searching and summarization work as expected.
- Performance: On a typical dev machine (4-core CPU, 16GB RAM), indexing 10–20 moderate-length documents completes in a few minutes. Summarization and QA are model-latency bound.
- Quality: The MiniLM embeddings produce good semantic neighbors for short to medium contexts. QA confidence threshold (0.3) helps filter low-confidence answers.

## 8. Limitations

- Large documents may exhaust memory when many embeddings are stored; FAISS + embeddings live in memory.
- Large-scale usage requires persistent vector DB (Milvus, Pinecone) and batching for embedding generation.
- Summarization and QA quality depend on model size and prompt truncation; very long context or complex reasoning tasks will be limited.

## 9. Future Work

- Add persistent vector store (FAISS on disk or Milvus/Pinecone)
- Add user authentication and per-user document stores
- Improve chunking using semantic splitters (sentence boundaries, overlap)
- Integrate async embedding batch processing for large uploads
- Add evaluation scripts and automatic tests for retrieval effectiveness

## 10. Conclusion

AI Document Brain is a practical, locally runnable tool to unlock documents using semantic embeddings, summaries and question answering. The recent improvements focused on usability, robustness and better feedback for end-users.

## Appendix A — Commands

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_cpu.txt
streamlit run app.py
```

## Appendix B — Dependencies

- streamlit
- sentence-transformers
- transformers
- torch
- faiss-cpu
- PyPDF2
