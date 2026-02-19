"""Insights — Narrated data visualizations."""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Insights", page_icon="📊", layout="wide")

st.title("📊 Data Insights")
st.markdown("Explore annotated visualizations revealing patterns in the Friends dataset.")
st.markdown("---")


@st.cache_data
def load_data():
    """Load processed dialogue data."""
    df = pd.read_csv("data/processed/dialogue_main_characters.csv")
    return df


df = load_data()

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dialogue Count",
    "📈 Season Trends",
    "📏 Speaking Style",
    "🔤 Top Words",
    "🥧 Dialogue Share",
])

from src.visualizations import (
    chart_dialogue_distribution,
    chart_dialogue_by_season,
    chart_avg_line_length,
    chart_top_words_per_character,
    chart_season_share,
)

with tab1:
    st.markdown("### Who Talks the Most?")
    st.markdown(
        "Rachel leads with **9,255 lines** across the entire series, "
        "closely followed by Ross. Phoebe, despite being a fan favorite, "
        "has the fewest lines among the main six."
    )
    chart = chart_dialogue_distribution(df)
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.markdown("### Dialogue Trends Across Seasons")
    st.markdown(
        "Track how each character's screen presence evolved. Notice how "
        "Rachel's lines peak in the later seasons as her career storyline "
        "takes center stage."
    )
    chart = chart_dialogue_by_season(df)
    st.altair_chart(chart, use_container_width=True)

with tab3:
    st.markdown("### Speaking Style: Words per Line")
    st.markdown(
        "Who gives speeches vs. who keeps it short? This reveals each "
        "character's conversational style — longer lines often indicate "
        "monologues, explanations, or emotional speeches."
    )
    chart = chart_avg_line_length(df)
    st.altair_chart(chart, use_container_width=True)

with tab4:
    st.markdown("### Most Frequent Words by Character")
    st.markdown(
        "After removing common stopwords, these are the distinctive words "
        "each character uses most. Look for patterns that reflect their "
        "personality traits and storylines."
    )
    chart = chart_top_words_per_character(df)
    st.altair_chart(chart, use_container_width=False)

with tab5:
    st.markdown("### Dialogue Share by Season")
    st.markdown(
        "A normalized view showing how screen time is distributed. "
        "The six friends share remarkably balanced screen time across "
        "most seasons — a testament to the ensemble writing."
    )
    chart = chart_season_share(df)
    st.altair_chart(chart, use_container_width=True)

# ── Fun Stats ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Quick Stats")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Lines", f"{len(df):,}")
with c2:
    st.metric("Episodes", f"{df['title'].nunique()}")
with c3:
    avg_words = df["cleaned_text"].str.split().str.len().mean()
    st.metric("Avg Words/Line", f"{avg_words:.1f}")
with c4:
    st.metric("Seasons", "10")
