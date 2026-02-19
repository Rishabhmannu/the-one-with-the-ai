"""
personality.py — Personality Matcher (Weighted Blend)

Three-signal approach for robust character matching:

1. SBERT (Sentence-BERT) — Context-aware sentence embeddings from
   'all-MiniLM-L6-v2'. Produces 384D vectors that understand full sentence
   meaning, not just word averages. Discriminative centroids.

2. Word2Vec — Our fine-tuned GloVe+Friends hybrid model. Captures
   Friends-specific vocabulary nuances. Discriminative centroids.

3. Topic Boosting — Curated keyword-to-character associations for
   character-defining topics (Monica=cooking, Ross=dinosaurs, etc.)

Final score = w1*SBERT + w2*W2V + w3*Topic
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.model import load_model, sentence_to_vector
from src.preprocessing import tokenize_and_clean

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_CHARACTERS = ["Rachel", "Ross", "Chandler", "Monica", "Joey", "Phoebe"]

# Blend weights (tunable)
WEIGHT_SBERT = 0.50
WEIGHT_W2V = 0.20
WEIGHT_TOPIC = 0.30

# Softmax temperature for percentage conversion
TEMPERATURE = 0.15

# ── Topic Boosting Keywords ──────────────────────────────────────────────────
# Curated character-topic associations based on show knowledge.
# Each keyword gets a weight that boosts the character's score when detected.
CHARACTER_TOPICS = {
    "Monica": {
        "keywords": [
            "clean", "cleaning", "organize", "organized", "tidy", "neat", "spotless",
            "cook", "cooking", "chef", "recipe", "food", "kitchen", "bake", "baking",
            "thanksgiving", "turkey", "restaurant", "competitive", "win", "winning",
            "control", "perfect", "perfectionist", "rules", "order",
            "hostess", "host", "dinner", "meal", "caterer",
        ],
        "weight": 1.0,
    },
    "Ross": {
        "keywords": [
            "dinosaur", "dinosaurs", "fossil", "fossils", "paleontology",
            "museum", "science", "scientist", "professor", "lecture",
            "history", "evolution", "research", "academic", "university",
            "phd", "doctor", "nerd", "nerdy", "intellectual",
            "divorce", "divorced", "marriage", "marriages", "wedding",
            "break", "we were on a break",
            "son", "ben", "emma",
        ],
        "weight": 1.0,
    },
    "Rachel": {
        "keywords": [
            "fashion", "clothes", "outfit", "dress", "style", "stylish",
            "designer", "design", "ralph lauren", "bloomingdale", "shopping",
            "shop", "shoes", "handbag", "purse", "accessories",
            "hair", "makeup", "magazine", "vogue", "trend", "trendy",
            "waitress", "coffee", "central perk",
            "spoiled", "daddy", "rich",
        ],
        "weight": 1.0,
    },
    "Chandler": {
        "keywords": [
            "sarcasm", "sarcastic", "joke", "jokes", "joking", "funny",
            "humor", "humorous", "irony", "ironic", "witty", "wit",
            "could this be", "could i be", "could there be",
            "awkward", "uncomfortable", "nervous", "anxiety",
            "commitment", "afraid", "scared",
            "data", "computer", "office", "work", "job", "boring",
            "smoking", "smoke", "cigarette",
            "janice", "oh my god",
        ],
        "weight": 1.0,
    },
    "Joey": {
        "keywords": [
            "act", "actor", "acting", "audition", "auditions", "role",
            "movie", "movies", "show", "television", "tv", "soap opera",
            "days of our lives", "dr. drake ramoray",
            "pizza", "sandwich", "sandwiches", "eat", "eating", "hungry",
            "food", "meat", "meatball",
            "women", "girl", "girls", "date", "dating", "hot",
            "how you doin", "how you doing",
            "dumb", "stupid", "confused",
        ],
        "weight": 1.0,
    },
    "Phoebe": {
        "keywords": [
            "song", "songs", "sing", "singing", "guitar", "music", "musician",
            "smelly cat", "cat", "cats",
            "massage", "masseuse", "therapy", "spiritual", "spirit",
            "vegan", "vegetarian", "animal", "animals", "nature",
            "weird", "strange", "quirky", "eccentric",
            "twin", "ursula", "mother", "grandmother",
            "psychic", "aura", "karma", "reincarnation",
            "homeless", "street", "tough",
        ],
        "weight": 1.0,
    },
}


import re as _re

def compute_topic_scores(text: str) -> dict:
    """
    Compute topic-based matching scores for each character.
    Returns dict: character -> score (0.0 to ~1.0)

    Uses sqrt scaling so even 1-2 keyword hits produce a meaningful score
    that doesn't get drowned out by the embedding signals.
    """
    text_lower = text.lower()
    # Strip punctuation from individual words so "paleontology," matches "paleontology"
    words = set(_re.sub(r"[^\w\s]", "", text_lower).split())

    scores = {}
    for character, config in CHARACTER_TOPICS.items():
        hits = 0
        for keyword in config["keywords"]:
            if " " in keyword:
                # Multi-word phrase: check in full cleaned text
                if keyword in _re.sub(r"[^\w\s]", "", text_lower):
                    hits += 1
            else:
                if keyword in words:
                    hits += 1

        # Sqrt scaling: 1 hit → 0.33, 2 hits → 0.47, 3 hits → 0.58, 5 hits → 0.75
        # This makes even a single keyword match meaningful
        if hits > 0:
            scores[character] = min(1.0, (hits / 3) ** 0.5) * config["weight"]
        else:
            scores[character] = 0.0

    return scores


class PersonalityMatcher:
    """Three-signal blended personality matcher."""

    def __init__(
        self,
        w2v_model,
        sbert_model,
        w2v_centroids: dict,
        w2v_global: np.ndarray,
        sbert_centroids: dict,
        sbert_global: np.ndarray,
    ):
        self.w2v_model = w2v_model
        self.sbert_model = sbert_model
        self.w2v_centroids = w2v_centroids
        self.w2v_global = w2v_global
        self.sbert_centroids = sbert_centroids
        self.sbert_global = sbert_global

    @classmethod
    def from_dataframe(cls, w2v_model, sbert_model, df: pd.DataFrame):
        """Build both SBERT and Word2Vec discriminative centroids."""
        print("🔧 Building character profiles (3-signal blend)...")

        # ── SBERT Centroids ───────────────────────────────────────────────
        print("\n📌 Signal 1: SBERT Centroids")
        sbert_raw_centroids = {}
        for character in MAIN_CHARACTERS:
            char_df = df[df["speaker"] == character]
            if char_df.empty:
                continue

            texts = char_df["cleaned_text"].tolist()
            # Encode in batches for efficiency
            embeddings = sbert_model.encode(texts, show_progress_bar=False, batch_size=256)
            centroid = np.mean(embeddings, axis=0)
            sbert_raw_centroids[character] = centroid
            print(f"  ✅ {character}: SBERT centroid from {len(texts):,} lines")

        # Discriminative SBERT centroids
        sbert_all = np.array(list(sbert_raw_centroids.values()))
        sbert_global = np.mean(sbert_all, axis=0)
        sbert_centroids = {}
        for char, raw in sbert_raw_centroids.items():
            direction = raw - sbert_global
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            sbert_centroids[char] = direction

        # Show SBERT inter-centroid similarities
        print("\n📊 SBERT distinctive direction similarities:")
        chars = list(sbert_centroids.keys())
        for i, c1 in enumerate(chars):
            for c2 in chars[i + 1:]:
                sim = cosine_similarity(
                    sbert_centroids[c1].reshape(1, -1),
                    sbert_centroids[c2].reshape(1, -1),
                )[0][0]
                print(f"  {c1} ↔ {c2}: {sim:.3f}")

        # ── Word2Vec Centroids ────────────────────────────────────────────
        print("\n📌 Signal 2: Word2Vec Discriminative Centroids")
        w2v_raw_centroids = {}
        for character in MAIN_CHARACTERS:
            char_df = df[df["speaker"] == character]
            if char_df.empty:
                continue

            vectors = []
            for text in char_df["cleaned_text"]:
                tokens = tokenize_and_clean(text)
                vec = sentence_to_vector(tokens, w2v_model)
                if np.linalg.norm(vec) > 0:
                    vectors.append(vec)
            if vectors:
                centroid = np.mean(vectors, axis=0)
                w2v_raw_centroids[character] = centroid
                print(f"  ✅ {character}: W2V centroid from {len(vectors):,} lines")

        w2v_all = np.array(list(w2v_raw_centroids.values()))
        w2v_global = np.mean(w2v_all, axis=0)
        w2v_centroids = {}
        for char, raw in w2v_raw_centroids.items():
            direction = raw - w2v_global
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            w2v_centroids[char] = direction

        print(f"\n📌 Signal 3: Topic Boosting ({sum(len(v['keywords']) for v in CHARACTER_TOPICS.values())} keywords)")
        print(f"\n✅ All profiles built. Blend weights: SBERT={WEIGHT_SBERT}, W2V={WEIGHT_W2V}, Topic={WEIGHT_TOPIC}")

        return cls(
            w2v_model=w2v_model,
            sbert_model=sbert_model,
            w2v_centroids=w2v_centroids,
            w2v_global=w2v_global,
            sbert_centroids=sbert_centroids,
            sbert_global=sbert_global,
        )

    def match(self, user_text: str) -> list:
        """
        Match user text to closest character using 3-signal blend.

        Signal 1: SBERT similarity (discriminative)
        Signal 2: Word2Vec similarity (discriminative)
        Signal 3: Topic keyword hits
        """
        if not user_text.strip():
            return []

        # ── Signal 1: SBERT ───────────────────────────────────────────────
        sbert_vec = self.sbert_model.encode([user_text])[0]
        sbert_disc = sbert_vec - self.sbert_global
        norm = np.linalg.norm(sbert_disc)
        if norm > 0:
            sbert_disc = sbert_disc / norm
        sbert_disc = sbert_disc.reshape(1, -1)

        sbert_scores = {}
        for char, centroid in self.sbert_centroids.items():
            sbert_scores[char] = cosine_similarity(sbert_disc, centroid.reshape(1, -1))[0][0]

        # ── Signal 2: Word2Vec ────────────────────────────────────────────
        tokens = tokenize_and_clean(user_text)
        w2v_vec = sentence_to_vector(tokens, self.w2v_model)
        w2v_disc = w2v_vec - self.w2v_global
        norm = np.linalg.norm(w2v_disc)
        if norm > 0:
            w2v_disc = w2v_disc / norm
        w2v_disc = w2v_disc.reshape(1, -1)

        w2v_scores = {}
        for char, centroid in self.w2v_centroids.items():
            w2v_scores[char] = cosine_similarity(w2v_disc, centroid.reshape(1, -1))[0][0]

        # ── Signal 3: Topic Boost ─────────────────────────────────────────
        topic_scores = compute_topic_scores(user_text)

        # ── Blend ─────────────────────────────────────────────────────────
        results = []
        for char in MAIN_CHARACTERS:
            if char not in sbert_scores:
                continue
            blended = (
                WEIGHT_SBERT * sbert_scores[char]
                + WEIGHT_W2V * w2v_scores.get(char, 0)
                + WEIGHT_TOPIC * topic_scores.get(char, 0)
            )
            results.append({
                "character": char,
                "blended_score": float(blended),
                "sbert": float(sbert_scores[char]),
                "w2v": float(w2v_scores.get(char, 0)),
                "topic": float(topic_scores.get(char, 0)),
            })

        results.sort(key=lambda x: x["blended_score"], reverse=True)

        # Softmax percentages on blended scores
        scores = np.array([r["blended_score"] for r in results])
        exp_s = np.exp((scores - scores.max()) / TEMPERATURE)
        percentages = exp_s / exp_s.sum() * 100
        for r, pct in zip(results, percentages):
            r["percentage"] = round(float(pct), 1)

        return results

    def match_multiple(self, user_texts: list) -> list:
        """Aggregate matching across multiple text inputs (quiz answers)."""
        aggregate = {char: 0.0 for char in MAIN_CHARACTERS}

        for text in user_texts:
            results = self.match(text)
            for r in results:
                aggregate[r["character"]] += r["blended_score"]

        n = len(user_texts) if user_texts else 1
        results = []
        for char, total in aggregate.items():
            results.append({"character": char, "blended_score": total / n})

        results.sort(key=lambda x: x["blended_score"], reverse=True)

        scores = np.array([r["blended_score"] for r in results])
        exp_s = np.exp((scores - scores.max()) / TEMPERATURE)
        percentages = exp_s / exp_s.sum() * 100
        for r, pct in zip(results, percentages):
            r["percentage"] = round(float(pct), 1)

        return results

    def get_winner(self, user_text: str) -> dict:
        """Get the best matching character."""
        results = self.match(user_text)
        return results[0] if results else None

    def save(self, path: str):
        """Save centroids and global vectors to disk."""
        data = {
            "w2v_centroids": {k: v.tolist() for k, v in self.w2v_centroids.items()},
            "w2v_global": self.w2v_global.tolist(),
            "sbert_centroids": {k: v.tolist() for k, v in self.sbert_centroids.items()},
            "sbert_global": self.sbert_global.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"💾 Profiles saved to {path}")

    @classmethod
    def load(cls, path: str, w2v_model, sbert_model):
        """Load centroids from disk."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            w2v_model=w2v_model,
            sbert_model=sbert_model,
            w2v_centroids={k: np.array(v) for k, v in data["w2v_centroids"].items()},
            w2v_global=np.array(data["w2v_global"]),
            sbert_centroids={k: np.array(v) for k, v in data["sbert_centroids"].items()},
            sbert_global=np.array(data["sbert_global"]),
        )


# ── Convenience ───────────────────────────────────────────────────────────────

def build_personality_matcher(
    w2v_model_path: str = "models/friends_w2v_hybrid.model",
    data_path: str = "data/processed/dialogue_main_characters.csv",
    profiles_path: str = "models/character_profiles.json",
    sbert_name: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
) -> PersonalityMatcher:
    """Build (or load cached) blended personality matcher."""
    w2v_model = load_model(w2v_model_path)
    sbert_model = SentenceTransformer(sbert_name)

    if os.path.exists(profiles_path) and not force_rebuild:
        matcher = PersonalityMatcher.load(profiles_path, w2v_model, sbert_model)
        print(f"✅ Loaded cached profiles from {profiles_path}")
    else:
        df = pd.read_csv(data_path)
        matcher = PersonalityMatcher.from_dataframe(w2v_model, sbert_model, df)
        matcher.save(profiles_path)

    return matcher


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_personality_demo():
    """Evaluate the blended personality matcher."""
    print("=" * 60)
    print("  Friends NLP — Personality Matcher (Weighted Blend)")
    print(f"  Weights: SBERT={WEIGHT_SBERT} | W2V={WEIGHT_W2V} | Topic={WEIGHT_TOPIC}")
    print("=" * 60)

    matcher = build_personality_matcher(force_rebuild=True)

    test_inputs = [
        ("🧹 Cleaning obsession",
         "Monica",
         "Oh my God, I need to clean this right now. This is unacceptable! Everything has to be organized and perfect."),
        ("🍕 Hungry for pizza",
         "Joey",
         "I'm so hungry right now! That pizza is calling my name! I want to eat everything!"),
        ("💔 On a break",
         "Ross",
         "We were on a break! It's not the same as being broken up! I need to explain the science of it."),
        ("🎸 Smelly Cat song",
         "Phoebe",
         "Oh I just wrote a new song about a cat named Smelly Cat who nobody wants to feed."),
        ("👗 Fashion & style",
         "Rachel",
         "Oh that outfit is so last season. Let me style you. I know fashion, I worked at Ralph Lauren."),
        ("🦕 Dinosaurs & science",
         "Ross",
         "Actually, the correct term is paleontology, not archaeology. Did you know the T-Rex had tiny arms?"),
        ("🎬 Acting audition",
         "Joey",
         "I'm an actor! I got a big audition today. I love sandwiches and could eat a whole pizza by myself."),
        ("😏 Sarcastic humor",
         "Chandler",
         "Could this day BE any worse? Oh sure, that sounds like a GREAT idea. What could possibly go wrong?"),
        ("🍗 Thanksgiving cooking",
         "Monica",
         "I'm going to make the best Thanksgiving turkey anyone has ever tasted. My food is the best!"),
        ("💘 Romantic fossil nerd",
         "Ross",
         "I brought you a fossil from the museum. Isn't it beautiful? I love dinosaurs and science so much."),
    ]

    correct = 0
    total = len(test_inputs)

    for scenario, expected, reaction in test_inputs:
        results = matcher.match(reaction)
        winner = results[0]
        match_ok = "✅" if winner["character"] == expected else "❌"
        if winner["character"] == expected:
            correct += 1

        print(f"\n{scenario} (expected: {expected})")
        print(f"   \"{reaction[:75]}...\"")
        print(f"   {match_ok} Winner: {winner['character']} ({winner['percentage']:.1f}%)")
        print(f"      Signals → SBERT={winner['sbert']:.3f}  W2V={winner['w2v']:.3f}  Topic={winner['topic']:.3f}")
        for r in results[:3]:
            bar = "█" * int(r["percentage"] / 2.5)
            print(f"      {r['character']:>10}: {bar} {r['percentage']:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_personality_demo()
