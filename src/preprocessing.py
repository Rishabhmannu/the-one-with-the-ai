"""
preprocessing.py — Text Cleaning & Tokenization Pipeline

Handles:
- Loading raw .txt script files from data/raw/
- Parsing dialogue lines (Speaker: Text format)
- Regex cleaning (scene/stage directions, speaker names)
- Tokenization
- Selective stopword removal (keeping sentiment words)
- Lemmatization via spaCy
- Saving processed data to data/processed/
"""

import os
import re
import pandas as pd
import spacy
import nltk

# Download NLTK stopwords (idempotent)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ── Stopwords Config ──────────────────────────────────────────────────────────
# Use NLTK's comprehensive stopword list (~198 words) as the base.
# Then ADD BACK sentiment/negation words that are important for personality.
_NLTK_STOPS = set(nltk_stopwords.words("english"))

# Words to KEEP even though NLTK marks them as stopwords
# (negation, sentiment modifiers that affect personality matching)
_KEEP_WORDS = {
    "not", "no", "nor", "very", "don", "don't",
    "didn", "didn't", "doesn", "doesn't",
    "isn", "isn't", "aren", "aren't",
    "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't",
    "couldn", "couldn't", "shouldn", "shouldn't",
    "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't",
    "against", "few", "more", "most",
}

STOP_WORDS_TO_REMOVE = _NLTK_STOPS - _KEEP_WORDS

# Known main characters (and common variations)
MAIN_CHARACTERS = {
    "monica", "ross", "rachel", "chandler", "joey", "phoebe",
    "rach", "mon", "pheebs", "chan",
}

# Lines to skip entirely
SKIP_LINES = {
    "commercial break", "opening credits", "closing credits",
    "end", "the end",
}

# Speaker names to skip (metadata, not dialogue)
SKIP_SPEAKERS = {
    "written by", "directed by", "teleplay by", "story by",
    "transcribed by", "note", "notes",
}


def parse_filename(filename: str) -> dict:
    """Extract season and episode info from a filename like 'S01E02 Title.txt'."""
    match = re.match(
        r"[Ss](\d{2})[Ee](\d{2})(?:-[Ss]?\d{2}[Ee](\d{2}))?\s+(.+)\.txt",
        filename,
    )
    if match:
        season = int(match.group(1))
        episode_start = int(match.group(2))
        episode_end = int(match.group(3)) if match.group(3) else episode_start
        title = match.group(4).strip()
        return {
            "season": season,
            "episode_start": episode_start,
            "episode_end": episode_end,
            "title": title,
        }
    return None


def is_scene_direction(line: str) -> bool:
    """Check if an entire line is a scene/stage direction."""
    stripped = line.strip()
    if not stripped:
        return True
    # Full line enclosed in brackets → scene direction
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    # Full line enclosed in parentheses → stage direction
    if stripped.startswith("(") and stripped.endswith(")"):
        return True
    return False


def clean_dialogue_text(text: str) -> str:
    """Clean a single dialogue line by removing inline stage directions."""
    # Remove inline parenthetical directions: (laughing), (to Ross), etc.
    text = re.sub(r"\([^)]*\)", "", text)
    # Remove inline bracketed directions: [Scene: ...]
    text = re.sub(r"\[[^\]]*\]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_dialogue_line(line: str) -> tuple:
    """
    Parse a line in 'Speaker: dialogue' format.
    Returns (speaker, text) or (None, None) if not a dialogue line.
    """
    # Match pattern: "SpeakerName: dialogue text"
    # Speaker names can contain spaces (e.g., "Dr. Ledbetter")
    # But the main pattern is "Word(s): text"
    match = re.match(r"^([A-Za-z][A-Za-z\s.'&,]+?):\s+(.+)", line)
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        # Filter out non-character speakers (scene labels, metadata, etc.)
        if speaker.lower() in SKIP_LINES or speaker.lower() in SKIP_SPEAKERS:
            return None, None
        return speaker, text
    return None, None


def normalize_speaker(speaker: str) -> str:
    """Normalize speaker names to canonical forms."""
    name = speaker.strip().lower()

    # Map common variations and multi-word speakers
    mapping = {
        "rach": "rachel",
        "rachel green": "rachel",
        "mon": "monica",
        "monica geller": "monica",
        "monica bing": "monica",
        "pheebs": "phoebe",
        "phoebe buffay": "phoebe",
        "chan": "chandler",
        "chandler bing": "chandler",
        "joey tribbiani": "joey",
        "ross geller": "ross",
    }

    if name in mapping:
        return mapping[name].title()

    # Capitalize first letter of each word
    return speaker.strip().title()


def load_scripts(data_dir: str) -> pd.DataFrame:
    """
    Load all .txt script files from data_dir and parse dialogue lines.
    Returns a DataFrame with columns:
        season, episode, title, speaker, raw_text
    """
    records = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue

        meta = parse_filename(filename)
        if meta is None:
            print(f"⚠️  Skipping file (unrecognized format): {filename}")
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty, scene directions, and meta lines
            if not line or is_scene_direction(line):
                continue
            if line.lower().strip() in SKIP_LINES:
                continue

            speaker, text = parse_dialogue_line(line)
            if speaker and text:
                text = clean_dialogue_text(text)
                if text:  # Skip if cleaning left nothing
                    records.append(
                        {
                            "season": meta["season"],
                            "episode": meta["episode_start"],
                            "title": meta["title"],
                            "speaker": normalize_speaker(speaker),
                            "raw_text": text,
                        }
                    )

    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df):,} dialogue lines from {df['title'].nunique()} episodes")
    return df


def tokenize_and_clean(
    text: str,
    stop_words: set = STOP_WORDS_TO_REMOVE,
    lemmatize: bool = True,
) -> list:
    """
    Tokenize and clean a text string.
    - Lowercases
    - Removes punctuation and numbers
    - Selective stopword removal
    - Lemmatization via spaCy
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # Skip punctuation, spaces, numbers
        if token.is_punct or token.is_space or token.like_num:
            continue
        # Skip single characters (except meaningful ones like "I")
        if len(token.text) <= 1 and token.text != "i":
            continue

        word = token.lemma_ if lemmatize else token.text

        # Selective stopword removal
        if word in stop_words:
            continue

        tokens.append(word)

    return tokens


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned/tokenized columns to the dataframe.
    - cleaned_text: cleaned string
    - tokens: list of cleaned tokens
    """
    print("🔄 Cleaning dialogue text...")
    df = df.copy()
    df["cleaned_text"] = df["raw_text"].apply(clean_dialogue_text)

    print("🔄 Tokenizing and lemmatizing (this may take a minute)...")
    df["tokens"] = df["cleaned_text"].apply(tokenize_and_clean)

    # Drop rows where tokenization produced empty lists
    before = len(df)
    df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"   Dropped {dropped} empty lines after tokenization")

    print(f"✅ Processed {len(df):,} dialogue lines")
    return df


def get_main_characters_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include the 6 main characters."""
    main = df[df["speaker"].str.lower().isin(MAIN_CHARACTERS)].copy()
    print(f"✅ Filtered to {len(main):,} lines from {main['speaker'].nunique()} main characters")
    return main


def save_processed_data(df: pd.DataFrame, output_dir: str):
    """Save processed dataframe to CSV and the token lists for Word2Vec."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full processed CSV
    csv_path = os.path.join(output_dir, "dialogue_processed.csv")
    df_save = df.drop(columns=["tokens"])  # tokens are lists, save separately
    df_save.to_csv(csv_path, index=False)
    print(f"💾 Saved processed CSV to {csv_path}")

    # Save tokens as one-sentence-per-line (space separated) for Word2Vec
    tokens_path = os.path.join(output_dir, "sentences.txt")
    with open(tokens_path, "w") as f:
        for tokens in df["tokens"]:
            f.write(" ".join(tokens) + "\n")
    print(f"💾 Saved sentence tokens to {tokens_path}")

    return csv_path, tokens_path


# ── Main Entry Point ──────────────────────────────────────────────────────────
def run_preprocessing(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
):
    """Full preprocessing pipeline."""
    print("=" * 60)
    print("  Friends NLP — Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load raw scripts
    df = load_scripts(raw_dir)
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"📊 Unique speakers: {df['speaker'].nunique()}")
    print(f"📊 Top speakers:\n{df['speaker'].value_counts().head(10)}")

    # 2. Process (clean + tokenize)
    df = process_dataframe(df)

    # 3. Save
    csv_path, tokens_path = save_processed_data(df, output_dir)

    # 4. Summary stats
    main_df = get_main_characters_df(df)
    print("\n📊 Main character line counts:")
    print(main_df["speaker"].value_counts().to_string())

    # Save main characters only version too
    main_csv = os.path.join(output_dir, "dialogue_main_characters.csv")
    main_df.drop(columns=["tokens"]).to_csv(main_csv, index=False)

    main_tokens = os.path.join(output_dir, "sentences_main.txt")
    with open(main_tokens, "w") as f:
        for tokens in main_df["tokens"]:
            f.write(" ".join(tokens) + "\n")

    print(f"\n💾 Saved main characters CSV to {main_csv}")
    print(f"💾 Saved main characters tokens to {main_tokens}")
    print("\n✅ Preprocessing complete!")

    return df, main_df


if __name__ == "__main__":
    run_preprocessing()
