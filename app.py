import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import io
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')
import logging

logging.getLogger('streamlit').setLevel(logging.ERROR)


@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
        return embedder, summarizer, qa_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else "Could not extract text from PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for better processing"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_vector_index(texts: List[str], embedder) -> Tuple[faiss.Index, np.ndarray]:
    """Create FAISS index from text embeddings"""
    if not texts:
        return None, None

    embeddings = embedder.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings


def semantic_search(query: str, index: faiss.Index, texts: List[str],
                    embedder, top_k: int = 5) -> List[Tuple[str, float]]:
    """Perform semantic search on document chunks"""
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, min(top_k, len(texts)))
    results = [(texts[i], float(distances[0][j]))
               for j, i in enumerate(indices[0]) if i < len(texts)]
    return results


def summarize_text(text: str, summarizer, max_length: int = 150) -> str:
    """Generate summary of text"""
    words = text.split()
    if len(words) < 50:
        return text

    try:
        text_truncated = " ".join(words[:1024])
        summary = summarizer(text_truncated,
                             max_length=max_length,
                             min_length=30,
                             do_sample=False,
                             truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {str(e)}"


def answer_question(question: str, context: str, qa_model) -> str:
    """Answer questions based on document context"""
    try:
        context_words = context.split()[:512]
        context_truncated = " ".join(context_words)

        result = qa_model(question=question, context=context_truncated)
        confidence = result['score']
        answer = result['answer']

        if confidence > 0.3:
            return f"**Answer:** {answer}\n\n*Confidence: {confidence:.2%}*"
        else:
            return "No confident answer found in the documents."
    except Exception as e:
        return f"Error answering question: {str(e)}"


st.set_page_config(
    page_title="AI Document Brain",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Document Brain: Semantic Search + Auto-Summaries")
st.markdown("Upload documents, build semantic search index, and query with natural language")

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'doc_names' not in st.session_state:
    st.session_state.doc_names = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.spinner("Loading AI models (this may take a minute on first run)..."):
        embedder, summarizer, qa_model = load_models()
        if embedder and summarizer and qa_model:
            st.session_state.embedder = embedder
            st.session_state.summarizer = summarizer
            st.session_state.qa_model = qa_model
            st.session_state.models_loaded = True
        else:
            st.error("Failed to load models. Please restart the application.")
            st.stop()

with st.sidebar:
    st.header(" Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDFs or Text files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more PDF or text files to build your document brain"
    )

    if st.button("🔨 Build Search Index", type="primary") and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            all_chunks = []
            doc_names = []

            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((idx) / len(uploaded_files))

                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file)
                else:
                    text = file.read().decode('utf-8', errors='ignore')

                if len(text.strip()) < 50:
                    st.warning(f" {file.name} has very little text content")
                    continue

                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                doc_names.extend([file.name] * len(chunks))

                st.session_state.documents.append({
                    'name': file.name,
                    'text': text
                })

            if all_chunks:
                status_text.text("Building vector index...")
                progress_bar.progress(0.9)

                st.session_state.chunks = all_chunks
                st.session_state.doc_names = doc_names
                st.session_state.index, _ = build_vector_index(
                    all_chunks,
                    st.session_state.embedder
                )

                progress_bar.progress(1.0)
                status_text.empty()
                st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents!")
            else:
                st.error("No valid text content found in uploaded files.")

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    if st.session_state.index:
        st.info(
            f" **Status:** {len(st.session_state.chunks)} chunks indexed from {len(st.session_state.documents)} documents")

    st.markdown("---")

    if st.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.doc_names = []
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs([" Semantic Search", " Summaries", " Q&A", " Documents"])

with tab1:
    st.header("Semantic Search")

    if st.session_state.index:
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., What are the main findings about climate change?",
                key="search_query"
            )

        with col2:
            top_k = st.slider("Results:", 1, 10, 5)

        if query:
            with st.spinner("Searching..."):
                results = semantic_search(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.embedder,
                    top_k
                )

            st.subheader(f"Top {len(results)} Results:")

            for i, (text, score) in enumerate(results, 1):
                with st.expander(f"**Result {i}** | Similarity Score: {1 / (1 + score):.3f}", expanded=(i == 1)):
                    st.write(text)
                    doc_idx = st.session_state.chunks.index(text)
                    st.caption(f" Source: **{st.session_state.doc_names[doc_idx]}**")
    else:
        st.info("Please upload documents and build the search index using the sidebar.")

with tab2:
    st.header("Document Summaries")

    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f" {doc['name']}"):
                col1, col2 = st.columns([1, 4])

                with col1:
                    if st.button("Generate Summary", key=f"sum_{doc['name']}"):
                        with st.spinner("Generating summary..."):
                            text_sample = " ".join(doc['text'].split()[:1000])
                            summary = summarize_text(
                                text_sample,
                                st.session_state.summarizer
                            )
                            st.session_state[f"summary_{doc['name']}"] = summary

                with col2:
                    if f"summary_{doc['name']}" in st.session_state:
                        st.markdown("**Summary:**")
                        st.info(st.session_state[f"summary_{doc['name']}"])
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload files.")

with tab3:
    st.header("Question Answering")

    if st.session_state.index:
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What is the main conclusion of the research?",
            key="qa_question"
        )

        if question:
            with st.spinner("Finding answer..."):
                results = semantic_search(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.embedder,
                    3
                )
                context = " ".join([r[0] for r in results])

                answer = answer_question(
                    question,
                    context,
                    st.session_state.qa_model
                )

            st.markdown("### Answer:")
            st.success(answer)

            with st.expander("View Context Used"):
                for i, (text, _) in enumerate(results, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(text[:500] + "...")
                    st.markdown("---")
    else:
        st.info(" Please upload documents and build the search index using the sidebar.")

with tab4:
    st.header("Uploaded Documents")

    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f" {doc['name']}"):
                word_count = len(doc['text'].split())
                char_count = len(doc['text'])

                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", f"{word_count:,}")
                col2.metric("Characters", f"{char_count:,}")
                col3.metric("Chunks", len([c for c in st.session_state.chunks if
                                           st.session_state.doc_names[st.session_state.chunks.index(c)] == doc[
                                               'name']]))

                st.text_area(
                    "Content Preview:",
                    doc['text'][:2000] + ("..." if len(doc['text']) > 2000 else ""),
                    height=300,
                    key=f"preview_{doc['name']}"
                )
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload files.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Tech Stack")
st.sidebar.markdown("""
- **Embeddings:** all-MiniLM-L6-v2
- **Vector Search:** FAISS
- **Summarization:** BART-Large-CNN
- **Q&A:** RoBERTa-SQuAD2
""")

st.sidebar.markdown("### Tips")
st.sidebar.markdown("""
- Upload multiple documents for richer search
- Use natural language queries
- Try specific questions for Q&A
- Summaries work best on longer docs
""")
