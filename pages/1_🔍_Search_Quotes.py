"""Search Quotes — Semantic search powered by Word2Vec."""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Search Quotes", page_icon="🔍", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .search-result {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        border-left: 4px solid #F5A623;
    }
    .result-speaker {
        font-weight: 700;
        font-size: 1rem;
        color: #F5A623;
    }
    .result-text {
        font-size: 1.05rem;
        color: #e0e0e0;
        margin: 6px 0;
        font-style: italic;
    }
    .result-meta {
        font-size: 0.8rem;
        color: #888;
    }
    .sim-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(80, 200, 120, 0.2);
        color: #50C878;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Search Quotes")
st.markdown("Search for Friends quotes by **meaning**, not just keywords.")


@st.cache_resource
def load_search_engine():
    """Load the semantic search engine (cached)."""
    from src.search import build_search_engine
    return build_search_engine()


# Load the search engine
with st.spinner("Loading search engine..."):
    engine = load_search_engine()

# ── Search UI ─────────────────────────────────────────────────────────────────
query = st.text_input(
    "What are you looking for?",
    placeholder="e.g., 'we were on a break', 'I love food', 'getting married'...",
    key="search_query",
)

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.selectbox("Results", [5, 10, 20], index=1)
with col2:
    speaker_filter = st.multiselect(
        "Filter by character",
        ["Rachel", "Ross", "Chandler", "Monica", "Joey", "Phoebe"],
        default=[],
    )

if query:
    results = engine.search(query, top_k=top_k * 2)  # get extras for filtering

    if speaker_filter:
        results = results[results["speaker"].isin(speaker_filter)]

    results = results.head(top_k)

    if results.empty:
        st.warning("No results found. Try a different query.")
    else:
        st.markdown(f"**{len(results)} results** for *\"{query}\"*")
        st.markdown("")

        for _, row in results.iterrows():
            sim_pct = row["similarity"] * 100
            color = "#50C878" if sim_pct >= 80 else "#F5A623" if sim_pct >= 60 else "#E8575A"
            st.markdown(
                f"""
                <div class="search-result">
                    <span class="result-speaker">{row['speaker']}</span>
                    <span class="sim-badge" style="background: rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.2); color: {color};">
                        {sim_pct:.0f}% match
                    </span>
                    <div class="result-text">"{row['raw_text']}"</div>
                    <div class="result-meta">
                        📺 Season {int(row['season'])}, Episode {int(row['episode'])} — {row['title']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    # Show example queries
    st.markdown("---")
    st.markdown("#### 💡 Try these searches:")
    examples = [
        "we were on a break",
        "I love food so much",
        "getting married",
        "I hate my job",
        "that's so funny",
    ]
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            st.code(example, language=None)
