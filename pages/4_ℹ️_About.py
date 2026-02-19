"""About — Project information, architecture, and credits."""
import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")
st.markdown("---")

# ── Overview ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    ### ☕ The One With The AI

    An end-to-end NLP project that uses **Word2Vec embeddings** and
    **Sentence-BERT** trained on Friends TV show scripts to power semantic
    search and personality matching.

    This project demonstrates the full NLP pipeline — from raw text preprocessing
    to deployed interactive features — using a fun, relatable dataset.
    """
)

# ── Architecture ──────────────────────────────────────────────────────────────
st.markdown("### 🏗️ Architecture")

st.markdown(
    """
    ```
    Raw Scripts (.txt)
        │
        ▼
    ┌─────────────────────────┐
    │   Preprocessing         │  spaCy + NLTK stopwords
    │   (cleaning, tokenizing,│  Selective stopword removal
    │    lemmatization)        │  Speaker normalization
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │   Word2Vec (Hybrid)     │  GloVe-300D pre-trained base
    │   Fine-tuned on         │  + 30 epochs fine-tuning
    │   Friends corpus        │  7,090 vocab, 93.3% coverage
    └────────┬────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
    ┌────────┐ ┌──────────────────┐
    │ Search │ │ Personality Match│
    │ (W2V)  │ │ (3-Signal Blend) │
    │        │ │ SBERT + W2V +    │
    │        │ │ Topic Boosting   │
    └────────┘ └──────────────────┘
        │              │
        └──────┬───────┘
               ▼
         ┌───────────┐
         │ Streamlit  │
         │ Frontend   │
         └───────────┘
    ```
    """
)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("### 🛠️ Technology Stack")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        | Layer | Technology |
        |---|---|
        | **Embedding Model** | Gensim Word2Vec (Skip-gram) |
        | **Pre-trained Base** | GloVe Wiki-Gigaword 300D |
        | **Sentence Embeddings** | Sentence-BERT (all-MiniLM-L6-v2) |
        | **Text Processing** | spaCy, NLTK |
        """
    )
with col2:
    st.markdown(
        """
        | Layer | Technology |
        |---|---|
        | **ML** | scikit-learn |
        | **Visualization** | Altair, PyNarrative |
        | **Frontend** | Streamlit |
        | **Data** | Pandas, NumPy |
        """
    )

# ── Key Design Decisions ─────────────────────────────────────────────────────
st.markdown("### 🎯 Key Design Decisions")

st.markdown(
    """
    1. **Hybrid Word2Vec** — Pure training on ~60K lines would produce weak embeddings.
       We start with GloVe (400K words) and fine-tune on Friends scripts for
       domain-specific vocabulary.

    2. **3-Signal Personality Blend** — Word2Vec centroids alone collapse because
       characters discuss similar topics. We combine:
       - **SBERT** (50%) — contextual sentence understanding
       - **Word2Vec** (20%) — Friends-specific patterns
       - **Topic Boosting** (30%) — curated character-keyword associations

    3. **NLTK Stopwords** — Comprehensive removal (~198 words) with sentiment words
       preserved (not, no, contractions) for personality matching accuracy.

    4. **Discriminative Centroids** — Instead of raw averaging (which causes centroid collapse),
       we subtract the global mean to isolate each character's *distinctive* direction.
    """
)

# ── Dataset ───────────────────────────────────────────────────────────────────
st.markdown("### 📊 Dataset Stats")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Dialogue Lines", "60,961")
with c2:
    st.metric("Main Character Lines", "49,208")
with c3:
    st.metric("Episodes", "228")
with c4:
    st.metric("Unique Speakers", "801")

# ── Credits ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    ### 📝 Credits

    - **Dataset**: Friends TV Show Script (Kaggle)
    - **Pre-trained Embeddings**: GloVe (Stanford NLP)
    - **Sentence Transformer**: all-MiniLM-L6-v2 (Hugging Face)
    - **Built by**: Rishabh

    ---
    *Made with ☕ and NLP*
    """
)
