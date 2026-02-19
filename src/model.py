"""
model.py — Word2Vec Training & Loading (Hybrid Approach)

Strategy:
1. Download pre-trained GloVe vectors (glove-wiki-gigaword-300) for robust
   general-purpose embeddings.
2. Initialize a Word2Vec model and load the pre-trained vectors.
3. Extend vocabulary with Friends-specific words from the corpus.
4. Fine-tune on the Friends corpus with a reduced learning rate to preserve
   general knowledge while absorbing domain-specific context.
5. Provide utilities for sentence vectorization and similarity.
"""

import os
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAINED_MODEL_NAME = "glove-wiki-gigaword-300"
VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 2
EPOCHS_FINETUNE = 30
LEARNING_RATE_START = 0.01   # Lower than default (0.025) to preserve pre-trained knowledge
LEARNING_RATE_MIN = 0.0001
WORKERS = 4


def load_sentences(sentences_path: str) -> list:
    """Load tokenized sentences from a text file (one sentence per line)."""
    sentences = []
    with open(sentences_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    print(f"✅ Loaded {len(sentences):,} sentences from {sentences_path}")
    return sentences


def download_pretrained() -> KeyedVectors:
    """Download pre-trained GloVe vectors via gensim downloader (cached after first download)."""
    print(f"📥 Loading pre-trained model: {PRETRAINED_MODEL_NAME}")
    print("   (First download is ~376 MB, cached for future use)")
    kv = api.load(PRETRAINED_MODEL_NAME)
    print(f"✅ Pre-trained model loaded: {len(kv):,} words, {kv.vector_size}D vectors")
    return kv


def build_hybrid_model(sentences: list, pretrained_kv: KeyedVectors) -> Word2Vec:
    """
    Build a hybrid Word2Vec model:
    1. Create a Word2Vec model and build vocabulary from the Friends corpus
    2. Intersect with pre-trained vectors (words in both get pre-trained values)
    3. Fine-tune on the Friends corpus
    """
    print("\n🔧 Building hybrid model...")

    # Step 1: Create Word2Vec model and build vocab from Friends corpus
    print("   Step 1/3: Building vocabulary from Friends corpus...")
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        alpha=LEARNING_RATE_START,
        min_alpha=LEARNING_RATE_MIN,
    )
    model.build_vocab(sentences)
    corpus_vocab_size = len(model.wv)
    print(f"   Corpus vocabulary: {corpus_vocab_size:,} words")

    # Step 2: Intersect with pre-trained vectors
    # Words found in both get the pre-trained vector as initialization
    # Words only in corpus get random initialization
    print("   Step 2/3: Intersecting with pre-trained vectors...")
    found = 0
    not_found = []
    for word in model.wv.key_to_index:
        if word in pretrained_kv:
            model.wv[word] = pretrained_kv[word]
            found += 1
        else:
            not_found.append(word)

    coverage = found / corpus_vocab_size * 100
    print(f"   Pre-trained coverage: {found:,}/{corpus_vocab_size:,} words ({coverage:.1f}%)")
    print(f"   Friends-specific words (new): {len(not_found):,}")
    if not_found[:20]:
        print(f"   Sample new words: {not_found[:20]}")

    # Step 3: Fine-tune on Friends corpus
    print(f"   Step 3/3: Fine-tuning on Friends corpus ({EPOCHS_FINETUNE} epochs)...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS_FINETUNE,
    )
    print("✅ Hybrid model training complete!")

    return model


def sentence_to_vector(
    sentence_tokens: list, model: Word2Vec, normalize: bool = True
) -> np.ndarray:
    """
    Convert a list of tokens to a sentence vector by averaging word vectors.
    Words not in vocabulary are skipped.
    Returns a zero vector if no words are found.
    """
    vectors = []
    for token in sentence_tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return np.zeros(model.wv.vector_size)

    vec = np.mean(vectors, axis=0)
    if normalize and np.linalg.norm(vec) > 0:
        vec = vec / np.linalg.norm(vec)
    return vec


def evaluate_model(model: Word2Vec):
    """Run sanity checks on the trained model."""
    print("\n🧪 Model Evaluation — Sanity Checks")
    print("=" * 50)

    # Most similar words
    test_words = ["coffee", "love", "food", "funny", "married", "baby", "apartment"]
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            similar_str = ", ".join([f"{w} ({s:.2f})" for w, s in similar])
            print(f"  '{word}' → {similar_str}")
        else:
            print(f"  '{word}' → ❌ not in vocabulary")

    # Character-pair similarities
    print(f"\n📊 Character Name Similarities:")
    characters = ["ross", "rachel", "monica", "chandler", "joey", "phoebe"]
    for i, c1 in enumerate(characters):
        for c2 in characters[i + 1 :]:
            if c1 in model.wv and c2 in model.wv:
                sim = model.wv.similarity(c1, c2)
                print(f"  {c1} ↔ {c2}: {sim:.3f}")

    # Vocabulary stats
    print(f"\n📊 Model Stats:")
    print(f"  Vocabulary size: {len(model.wv):,}")
    print(f"  Vector dimensions: {model.wv.vector_size}")


def save_model(model: Word2Vec, model_dir: str = "models"):
    """Save the trained model to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "friends_w2v_hybrid.model")
    model.save(model_path)
    print(f"\n💾 Model saved to {model_path}")

    # Also save just the KeyedVectors for lighter loading
    kv_path = os.path.join(model_dir, "friends_w2v_hybrid.kv")
    model.wv.save(kv_path)
    print(f"💾 KeyedVectors saved to {kv_path}")

    return model_path, kv_path


def load_model(model_path: str = "models/friends_w2v_hybrid.model") -> Word2Vec:
    """Load a previously saved model."""
    model = Word2Vec.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    print(f"   Vocabulary: {len(model.wv):,} words, {model.wv.vector_size}D")
    return model


# ── Main Entry Point ──────────────────────────────────────────────────────────
def run_training(
    sentences_path: str = "data/processed/sentences.txt",
    model_dir: str = "models",
):
    """Full training pipeline."""
    print("=" * 60)
    print("  Friends NLP — Word2Vec Hybrid Training Pipeline")
    print("=" * 60)

    # 1. Load corpus sentences
    sentences = load_sentences(sentences_path)

    # 2. Download/load pre-trained vectors
    pretrained_kv = download_pretrained()

    # 3. Build hybrid model
    model = build_hybrid_model(sentences, pretrained_kv)

    # 4. Evaluate
    evaluate_model(model)

    # 5. Save
    save_model(model, model_dir)

    print("\n✅ Training pipeline complete!")
    return model


if __name__ == "__main__":
    run_training()
