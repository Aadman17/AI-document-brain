# AI Document Brain

Lightweight Streamlit app for semantic document search, summarization and Q&A using sentence-transformers, FAISS and Hugging Face pipelines.

## Key Improvements

- Suppresses the warning - Added logging configuration to filter out Streamlit warnings
- Better error handling - Try-catch blocks throughout with helpful error messages
- Progress indicators - Visual feedback during document processing
- Enhanced UI - Better layout with metrics, status indicators, and organized tabs
- Memory efficiency - Models cached and loaded once, progress bars for long operations
- Clear document button - Easy way to reset and start over

## New Features Added

- Similarity scores displayed in search results
- Document statistics (word count, character count, chunk count)
- Expandable context view in Q&A showing which chunks were used
- Confidence scores for question answers
- Progress bar during index building
- Tips section in sidebar

## Features

- Upload PDFs and text files
- Build a FAISS vector index of document chunks
- Semantic search over chunks using sentence-transformers embeddings
- Per-document summarization using a Hugging Face summarization pipeline
- Question-answering over indexed documents using a SQuAD-style QA model
- UI tabs for Search, Summaries, Q&A, and uploaded Documents

## Tech Stack

- Python 3.8+
- Streamlit
- sentence-transformers (all-MiniLM-L6-v2)
- transformers (Hugging Face — BART, RoBERTa)
- FAISS
- PyPDF2

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```

2. Install dependencies (CPU):

```bash
pip install -r requirements_cpu.txt
```

Or for GPU:

```bash
pip install -r requirements_gpu.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser after starting with `streamlit run app.py`.
2. Upload one or more PDF or text files in the sidebar.
3. Click **Build Search Index** to extract text, chunk it, compute embeddings and build a FAISS index (progress bar will show).
4. Use the **Semantic Search** tab to run queries; similarity scores are shown.
5. Use **Summaries** to generate per-document summaries.
6. Use **Q&A** to ask targeted questions — the app will show an answer and a confidence score and allow you to inspect the context chunks used.
7. Use **Uploaded Documents** tab to view stats and previews. Use **Clear All Documents** to reset.

## Troubleshooting

- If you see transformer/accelerate warnings like `missing ScriptRunContext`, set the environment variables or adjust package versions (see project report).
- If models fail to load, check internet connectivity (initial model download) and available disk space.
- For memory issues, use smaller models or run on a machine with more RAM.

## License

This project is provided as-is for educational and prototyping use.
