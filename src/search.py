"""
search.py - Semantic Search Logic

Handles:
- Building the search index (sentence vectors for all dialogue lines)
- Query vectorization using the same preprocessing pipeline
- Cosine similarity ranking
- Top-K result retrieval with metadata
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.model import load_model, sentence_to_vector
from src.preprocessing import tokenize_and_clean


class SemanticSearch:
    """Semantic search engine over Friends dialogue lines."""

    def __init__(self, model, df: pd.DataFrame, vectors: np.ndarray = None):
        """
        Args:
            model: Trained Word2Vec model
            df: DataFrame with columns [season, episode, title, speaker, raw_text, cleaned_text]
            vectors: Pre-computed sentence vectors (optional, computed if None)
        """
        self.model = model
        self.df = df.reset_index(drop=True)
        self.vectors = vectors if vectors is not None else self._build_index()

    def _build_index(self) -> np.ndarray:
        """Build sentence vectors for all dialogue lines."""
        print("🔧 Building search index...")
        vectors = []
        for text in self.df["cleaned_text"]:
            tokens = tokenize_and_clean(text)
            vec = sentence_to_vector(tokens, self.model)
            vectors.append(vec)

        self.vectors = np.array(vectors, dtype=np.float32)
        print(f"✅ Index built: {self.vectors.shape[0]:,} vectors of dim {self.vectors.shape[1]}")
        return self.vectors

    def search(self, query: str, top_k: int = 10, min_words: int = 3) -> pd.DataFrame:
        """
        Search for quotes semantically similar to the query.

        Args:
            query: User's search text
            top_k: Number of results to return
            min_words: Minimum words in a dialogue line to be considered a result

        Returns:
            DataFrame with columns: [speaker, raw_text, season, episode, title, similarity]
        """
        # Tokenize and vectorize the query
        query_tokens = tokenize_and_clean(query)
        if not query_tokens:
            return pd.DataFrame()

        query_vec = sentence_to_vector(query_tokens, self.model).reshape(1, -1)

        # Skip zero vectors (empty query)
        if np.linalg.norm(query_vec) == 0:
            return pd.DataFrame()

        # Compute similarities
        sims = cosine_similarity(query_vec, self.vectors).flatten()

        # Create results dataframe
        results = self.df.copy()
        results["similarity"] = sims

        # Filter: only lines with enough words to be meaningful
        results["word_count"] = results["cleaned_text"].str.split().str.len()
        results = results[results["word_count"] >= min_words]

        # Sort by similarity (descending) and take top-K
        results = results.nlargest(top_k, "similarity")

        # Select display columns
        results = results[
            ["speaker", "raw_text", "season", "episode", "title", "similarity"]
        ].reset_index(drop=True)

        return results

    def save_index(self, path: str):
        """Save the pre-computed vectors to disk."""
        np.save(path, self.vectors)
        print(f"💾 Search index saved to {path}")

    @classmethod
    def load_index(cls, path: str, model, df: pd.DataFrame):
        """Load pre-computed vectors from disk."""
        vectors = np.load(path)
        print(f"✅ Search index loaded: {vectors.shape}")
        return cls(model=model, df=df, vectors=vectors)


# ── Convenience Functions ─────────────────────────────────────────────────────

def build_search_engine(
    model_path: str = "models/friends_w2v_hybrid.model",
    data_path: str = "data/processed/dialogue_processed.csv",
    index_path: str = "models/search_index.npy",
) -> SemanticSearch:
    """Build (or load cached) search engine."""
    import os

    model = load_model(model_path)
    df = pd.read_csv(data_path)

    if os.path.exists(index_path):
        engine = SemanticSearch.load_index(index_path, model, df)
    else:
        engine = SemanticSearch(model=model, df=df)
        engine.save_index(index_path)

    return engine


# ── Main Entry Point (for testing) ───────────────────────────────────────────

def run_search_demo():
    """Interactive search demo."""
    print("=" * 60)
    print("  Friends NLP - Semantic Search Demo")
    print("=" * 60)

    engine = build_search_engine()

    test_queries = [
        "I'm so hungry",
        "I want to get married",
        "that's so funny",
        "I hate my job",
        "we were on a break",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 50)
        results = engine.search(query, top_k=5)
        for i, row in results.iterrows():
            print(
                f"  {i+1}. [{row['speaker']}] \"{row['raw_text'][:80]}...\" "
                f"(S{row['season']:02d}E{row['episode']:02d}, sim={row['similarity']:.3f})"
            )


if __name__ == "__main__":
    run_search_demo()
