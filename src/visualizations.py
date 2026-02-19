"""
visualizations.py — Data Visualization Builders

Creates Altair charts for the Streamlit Insights page.
Uses a consistent Friends character color palette.
"""

import altair as alt
import pandas as pd
import numpy as np


# ── Character color palette ───────────────────────────────────────────────────
CHARACTER_COLORS = {
    "Rachel": "#E8575A",
    "Ross": "#5B8DB8",
    "Chandler": "#F5A623",
    "Monica": "#7B68EE",
    "Joey": "#50C878",
    "Phoebe": "#FF69B4",
}

COLOR_SCALE = alt.Scale(
    domain=list(CHARACTER_COLORS.keys()),
    range=list(CHARACTER_COLORS.values()),
)


def chart_dialogue_distribution(df: pd.DataFrame) -> alt.Chart:
    """
    Horizontal bar chart showing total dialogue lines per main character.
    """
    counts = (
        df.groupby("speaker")
        .size()
        .reset_index(name="lines")
        .sort_values("lines", ascending=True)
    )

    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=6, size=28)
        .encode(
            x=alt.X("lines:Q", title="Number of Lines", axis=alt.Axis(grid=False)),
            y=alt.Y("speaker:N", title=None, sort="-x"),
            color=alt.Color("speaker:N", scale=COLOR_SCALE, legend=None),
            tooltip=[
                alt.Tooltip("speaker:N", title="Character"),
                alt.Tooltip("lines:Q", title="Lines", format=","),
            ],
        )
    )

    text = chart.mark_text(
        align="left", dx=5, fontSize=13, fontWeight="bold"
    ).encode(text=alt.Text("lines:Q", format=","))

    return (chart + text).properties(
        width=550,
        height=350,
        title=alt.TitleParams(
            "Who Talks the Most?",
            subtitle="Total dialogue lines across all 10 seasons",
        ),
    )


def chart_dialogue_by_season(df: pd.DataFrame) -> alt.Chart:
    """
    Line chart showing dialogue count per character across seasons.
    """
    season_counts = (
        df.groupby(["season", "speaker"])
        .size()
        .reset_index(name="lines")
    )

    chart = (
        alt.Chart(season_counts)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("lines:Q", title="Number of Lines"),
            color=alt.Color("speaker:N", scale=COLOR_SCALE, title="Character"),
            tooltip=[
                alt.Tooltip("speaker:N", title="Character"),
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("lines:Q", title="Lines", format=","),
            ],
        )
        .properties(
            width=600,
            height=380,
            title=alt.TitleParams(
                "Dialogue Trends Across Seasons",
                subtitle="How each character's screen presence evolved over 10 seasons",
            ),
        )
    )

    return chart


def chart_avg_line_length(df: pd.DataFrame) -> alt.Chart:
    """
    Bar chart showing average words per dialogue line for each character.
    """
    df = df.copy()
    df["word_count"] = df["cleaned_text"].str.split().str.len()
    avg_len = (
        df.groupby("speaker")["word_count"]
        .mean()
        .reset_index(name="avg_words")
        .sort_values("avg_words", ascending=True)
    )
    avg_len["avg_words"] = avg_len["avg_words"].round(1)

    chart = (
        alt.Chart(avg_len)
        .mark_bar(cornerRadiusEnd=6, size=28)
        .encode(
            x=alt.X("avg_words:Q", title="Average Words per Line"),
            y=alt.Y("speaker:N", title=None, sort="-x"),
            color=alt.Color("speaker:N", scale=COLOR_SCALE, legend=None),
            tooltip=[
                alt.Tooltip("speaker:N", title="Character"),
                alt.Tooltip("avg_words:Q", title="Avg Words"),
            ],
        )
    )

    text = chart.mark_text(
        align="left", dx=5, fontSize=13, fontWeight="bold"
    ).encode(text=alt.Text("avg_words:Q", format=".1f"))

    return (chart + text).properties(
        width=550,
        height=350,
        title=alt.TitleParams(
            "Speaking Style: Words per Line",
            subtitle="Who gives speeches vs. who keeps it short?",
        ),
    )


def chart_top_words_per_character(df: pd.DataFrame, top_n: int = 8) -> alt.Chart:
    """
    Faceted bar chart showing the most frequent words per character.
    Uses the already-cleaned text to avoid slow spaCy re-tokenization.
    """
    from collections import Counter
    from src.preprocessing import STOP_WORDS_TO_REMOVE

    # Combine standard stopwords with visualization-specific ones
    # We remove these ONLY for the chart to surface distinctive vocabulary.
    extra_stops = {
        # Common conversational fillers & verbs
        "know", "go", "get", "oh", "yeah", "well", "okay", "right", "hey",
        "come", "say", "want", "think", "tell", "look", "see", "take", "make",
        "like", "good", "really", "one", "thing", "guy", "mean", "yes", "let",
        "time", "gonna", "would", "could", "should", "great", "sorry", "thank",
        "god", "wait", "back", "little", "never", "love", "much", "even",
        
        # Negations & Contractions (kept for model, removed for viz)
        "not", "no", "dont", "cant", "wont", "didnt", "wouldnt", "couldnt",
        "im", "ive", "id", "ill", "hes", "shes", "theyre", "thats", "whats",
        "theres", "isnt", "arent", "aint",
        "don't", "can't", "won't", "didn't", "wouldn't", "couldn't",
        "i'm", "i've", "i'd", "i'll", "he's", "she's", "they're", "that's",
        "what's", "there's", "isn't", "aren't", "ain't", "it's", "we're",
        "you're", "you've", "you'll", "you'd", "who's", "here's",

        # Curly quotes versions (U+2019)
        "don’t", "can’t", "won’t", "didn’t", "wouldn’t", "couldn’t",
        "i’m", "i’ve", "i’d", "i’ll", "he’s", "she’s", "they’re", "that’s",
        "what’s", "there’s", "isn’t", "aren’t", "ain’t", "it’s", "we’re",
        "you’re", "you’ve", "you’ll", "you’d", "who’s", "here’s",
        "y'know", "y’know",

        # Pronoun/Auxiliary artifacts & Generic words
        "re", "ve", "ll", "d", "m", "s", "t", "us", "got", "give",
        "going", "getting", "doing", "saying", "talking", "taking", "coming",
        "wanna", "gonna", "gotta", "lemme",
        "thing", "things", "something", "anything", "nothing", "everything",
        "maybe", "actually", "probably", "totally", "literally", "seriously",
        "guy", "guys", "man", "woman", "men", "women",
        "call", "work", "home", "room", 
        
        # Character names (they talk about each other a lot, but it's not a "catchphrase")
        "ross", "rachel", "monica", "joey", "chandler", "phoebe",
    }
    
    stop_words = STOP_WORDS_TO_REMOVE.union(extra_stops)

    records = []
    for character in ["Rachel", "Ross", "Chandler", "Monica", "Joey", "Phoebe"]:
        char_df = df[df["speaker"] == character]
        word_counts = Counter()
        for text in char_df["cleaned_text"].dropna():
            # Use already-cleaned text — just split on whitespace
            for word in text.lower().split():
                word = word.strip(".,!?;:'\"()-")
                if word and word not in stop_words and len(word) > 2:
                    word_counts[word] += 1

        for word, count in word_counts.most_common(top_n):
            records.append({"character": character, "word": word, "count": count})

    word_df = pd.DataFrame(records)

    chart = (
        alt.Chart(word_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title="Frequency"),
            y=alt.Y("word:N", title=None, sort="-x"),
            color=alt.Color("character:N", scale=COLOR_SCALE, legend=None),
            tooltip=["character", "word", "count"],
        )
        .properties(width=180, height=200)
        .facet("character:N", columns=3)
        .resolve_scale(y="independent")
        .properties(title="Most Frequent Words by Character")
    )

    return chart


def chart_season_share(df: pd.DataFrame) -> alt.Chart:
    """
    Stacked area chart showing each character's share of dialogue per season.
    """
    season_counts = (
        df.groupby(["season", "speaker"])
        .size()
        .reset_index(name="lines")
    )

    totals = season_counts.groupby("season")["lines"].transform("sum")
    season_counts["percentage"] = (season_counts["lines"] / totals * 100).round(1)

    chart = (
        alt.Chart(season_counts)
        .mark_area(opacity=0.8)
        .encode(
            x=alt.X("season:O", title="Season", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("percentage:Q", title="Share of Dialogue (%)", stack="normalize"),
            color=alt.Color("speaker:N", scale=COLOR_SCALE, title="Character"),
            tooltip=[
                alt.Tooltip("speaker:N", title="Character"),
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("percentage:Q", title="Share %", format=".1f"),
            ],
        )
        .properties(
            width=600,
            height=380,
            title=alt.TitleParams(
                "Dialogue Share by Season",
                subtitle="How screen time shifted across the series",
            ),
        )
    )

    return chart
