import streamlit as st

st.set_page_config(
    page_title="The One With The AI",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        font-family: 'Outfit', sans-serif;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #E8575A, #F5A623, #7B68EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .feature-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    .feature-icon {
        font-size: 2.4rem;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 8px;
    }

    .feature-desc {
        font-size: 0.95rem;
        color: #aaa;
        line-height: 1.4;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #F5A623;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .tech-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 3px;
        background: rgba(123, 104, 238, 0.15);
        color: #7B68EE;
        border: 1px solid rgba(123, 104, 238, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">☕ The One With The AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Explore the Friends universe through the lens of NLP</p>',
    unsafe_allow_html=True,
)

# ── Feature Cards ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Semantic Search</div>
            <div class="feature-desc">
                Search for Friends quotes by <strong>meaning</strong>, not just keywords.
                Powered by Word2Vec embeddings fine-tuned on 60K+ dialogue lines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Personality Quiz</div>
            <div class="feature-desc">
                React to scenarios and discover which Friend matches your personality.
                Uses SBERT + Word2Vec + topic analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Data Insights</div>
            <div class="feature-desc">
                Explore narrated visualizations revealing patterns in
                character dialogue, speaking styles, and series trends.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Stats Row ─────────────────────────────────────────────────────────────────
st.markdown("---")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<p class="stat-number">60,961</p><p class="stat-label">Dialogue Lines</p>', unsafe_allow_html=True)
with s2:
    st.markdown('<p class="stat-number">228</p><p class="stat-label">Episodes</p>', unsafe_allow_html=True)
with s3:
    st.markdown('<p class="stat-number">10</p><p class="stat-label">Seasons</p>', unsafe_allow_html=True)
with s4:
    st.markdown('<p class="stat-number">3</p><p class="stat-label">NLP Models</p>', unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🛠️ Technology Stack")
techs = [
    "Word2Vec (GloVe)", "Sentence-BERT", "spaCy", "NLTK",
    "scikit-learn", "Altair", "PyNarrative", "Streamlit", "Gensim",
]
badges = " ".join([f'<span class="tech-badge">{t}</span>' for t in techs])
st.markdown(badges, unsafe_allow_html=True)

st.markdown("")
st.markdown("👈 **Use the sidebar** to navigate between features.")
