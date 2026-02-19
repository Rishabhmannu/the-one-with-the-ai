# 📋 Project Plan: "The One With The AI"
## End-to-End NLP Project with Friends TV Show Dataset

---

## 1. Project Overview

**Goal:** Build an interactive, production-quality web application that lets users explore the Friends TV show universe using Natural Language Processing (NLP).

**Core Application Features:**
1. **Semantic Search Engine** — Search for quotes by *meaning*, not just keywords
2. **Personality Matcher Quiz** — Match a user's personality to a Friends character based on their text input
3. **Data Insights Dashboard** — Narrated, annotated visualizations exploring the dataset (powered by PyNarrative)

**Post-Project Deliverables:**
- GitHub repository with clean documentation
- Animated data storytelling walkthrough (powered by ipyvizzu-story)
- Medium blog article(s)

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core programming language |
| **NLP Model** | Gensim Word2Vec (Skip-gram) | Domain-specific word embeddings trained on Friends scripts |
| **Preprocessing** | spaCy / NLTK | Tokenization, lemmatization, stopword handling |
| **Similarity** | scikit-learn (cosine_similarity) | Vector comparison for search & personality matching |
| **Frontend** | Streamlit | Interactive web application |
| **In-App Viz** | PyNarrative (Altair-based) | Narrated, annotated charts in the Insights page |
| **Blog/Docs Viz** | ipyvizzu-story | Animated data story presentations for Medium/GitHub |
| **Data** | Pandas | Data manipulation & cleaning |
| **Deployment** | Streamlit Cloud | Free hosting via GitHub integration |
| **Version Control** | Git + GitHub | Source code management |

---

## 3. Data Requirements

### A. Text Data: Friends TV Scripts
- **Source:** Kaggle — Friends TV Show Script dataset
- **Required Columns:** `Season`, `Episode`, `Scene`, `Speaker`, `Text`
- **Volume:** ~10 seasons, ~236 episodes (~61,000+ dialogue lines)
- **Storage:** Fits entirely in RAM (< 50 MB)

### B. Image Assets
- **Character Avatars (6):** One image per main character (Monica, Joey, Chandler, Ross, Rachel, Phoebe)
  - Source: Royalty-free images or generated assets
  - Location: `assets/characters/`
- **Quiz Scenario Images (3–5):** Trigger images for the personality quiz
  - Examples: Messy room (Monica), large pizza (Joey), breakup scene (Ross), guitar (Phoebe), fashion items (Rachel), sarcastic sign (Chandler)
  - Source: Unsplash / Pexels (royalty-free)
  - Location: `assets/scenarios/`

---

## 4. Project Structure

```
nlp-project/
├── PROJECT_PLAN.md              # This file
├── requirements.txt             # Python dependencies
├── README.md                    # GitHub README with project overview
├── app.py                       # Streamlit entry point
│
├── data/
│   ├── raw/                     # Original Kaggle CSV(s)
│   └── processed/               # Cleaned, tokenized data
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Text cleaning & tokenization pipeline
│   ├── model.py                 # Word2Vec training & loading
│   ├── search.py                # Semantic search logic
│   ├── personality.py           # Personality matcher logic
│   └── visualizations.py        # PyNarrative chart builders
│
├── models/
│   └── friends_word2vec.model   # Trained Word2Vec model
│
├── assets/
│   ├── characters/              # Character avatar images
│   └── scenarios/               # Quiz scenario images
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Preprocessing walkthrough
│   ├── 03_model_training.ipynb  # Word2Vec training & evaluation
│   └── 04_data_story.ipynb      # ipyvizzu-story animated presentation
│
├── pages/                       # Streamlit multi-page app
│   ├── 1_🔍_Search_Quotes.py
│   ├── 2_🎭_Personality_Quiz.py
│   ├── 3_📊_Insights.py
│   └── 4_ℹ️_About.py
│
└── docs/
    └── blog_draft.md            # Medium article draft
```

---

## 5. Pre-processing Pipeline (preprocessing.py)

The quality of the NLP model depends entirely on the quality of cleaning. This is the "secret sauce."

### Step-by-Step:

1. **Load Data**
   - Read CSV into Pandas DataFrame
   - Verify required columns: `Season`, `Episode`, `Scene`, `Speaker`, `Text`

2. **Regex Cleaning**
   - Remove stage/scene directions: `[Scene: Central Perk]`, `(Jumping on the bed)`, etc.
   - Remove speaker name prefixes if embedded in text lines
   - Strip special characters, extra whitespace

3. **Tokenization**
   - Split dialogue into individual words using spaCy or NLTK tokenizer

4. **Normalization**
   - Convert all text to lowercase
   - **Selective stopword removal** — This is critical:
     - ✅ Remove pure noise words: `the`, `a`, `an`, `is`, `are`, `was`, `were`
     - ❌ Keep sentiment/personality words: `not`, `very`, `hate`, `love`, `never`, `always`
     - Rationale: The personality matcher relies on emotional/sentiment cues
   - **Lemmatization**: Convert inflected forms to base (`running` → `run`, `better` → `good`)
     - Use spaCy's lemmatizer for accuracy

5. **Output**
   - List of cleaned sentences (list of lists of words) → ready for Word2Vec
   - Cleaned DataFrame with original metadata preserved → for search results display
   - Save processed data to `data/processed/`

---

## 6. Model Architecture & Training (model.py)

### Primary Model: Gensim Word2Vec

| Hyperparameter | Value | Rationale |
|---|---|---|
| `sg` | `1` (Skip-gram) | Better performance on smaller datasets |
| `vector_size` | `150` (start), tune between 100–300 | Balance between expressiveness and overfitting |
| `window` | `5` | Look 5 words in each direction for context |
| `min_count` | `2` | Ignore words that appear only once |
| `epochs` | `30` | More epochs needed due to small dataset |
| `workers` | `4` | Leverage M4 Pro multi-core |

### Training Process:
1. Feed the cleaned list of sentences into `Word2Vec()`
2. Save trained model to `models/friends_word2vec.model`
3. **Sanity checks** after training:
   - `model.wv.most_similar('coffee')` → expect "Central Perk" related words
   - `model.wv.most_similar('love')` → expect emotional/relationship terms
   - `model.wv.most_similar('food')` → expect Joey-related terms
   - `model.wv.similarity('ross', 'rachel')` → expect high similarity

### Fallback Strategy:
- If the custom vocabulary is too small or embeddings are poor quality, use a **pre-trained model** (e.g., Google News Word2Vec or GloVe) as a supplement
- Can also combine: custom embeddings for Friends-specific terms + pre-trained for general terms

---

## 7. Feature Implementation

### Feature A: Semantic Search (search.py)

#### Offline — Build the Search Index:
1. Take every cleaned dialogue line from the dataset
2. Convert each line to a **Sentence Vector** by averaging the Word2Vec vectors of its constituent words
   - Handle unknown words by skipping them
   - Handle empty vectors gracefully (fallback to zero vector)
3. Store as a searchable structure:
   ```
   [
     {"text": "Joey doesn't share food", "speaker": "Joey", "season": 5, "episode": 12, "vector": [0.12, -0.4, ...]},
     ...
   ]
   ```

#### Real-time — Search Logic:
1. User types query (e.g., "I want a sandwich")
2. Clean and tokenize the query using the same preprocessing pipeline
3. Convert query to a vector (average of Word2Vec vectors)
4. Calculate **Cosine Similarity** between query vector and all stored vectors
5. Return **top 5–10 results**, ranked by similarity score
6. Display results as cards showing: quote, character name, season/episode

---

### Feature B: Personality Matcher (personality.py)

#### Offline — Build Character Profiles:
1. **Filter dialogue by speaker** — separate all lines for each of the 6 main characters:
   - Monica, Joey, Chandler, Ross, Rachel, Phoebe
2. **Vectorize each line** using the same averaging approach
3. **Compute the Centroid** (element-wise average) of all line vectors per character
   - This centroid IS the "Character Vector" — it represents the aggregate semantic fingerprint of how that character speaks
4. Store the 6 centroid vectors for real-time matching

#### Real-time — Matching Logic:
1. User sees a scenario image (e.g., messy room)
2. User types their reaction (e.g., "It makes me anxious, I need to clean it.")
3. Preprocess and vectorize the user's input
4. Calculate **Cosine Similarity** between the user's vector and each of the 6 character centroids
5. **Winner** = character with highest similarity score
6. Display results:
   - Character avatar
   - Similarity percentages for all 6 characters (progress bars)
   - Fun description of the matching character

---

### Feature C: Data Insights Dashboard (visualizations.py + PyNarrative)

This is the **new addition** to the original blueprint — leveraging **PyNarrative** to create annotated, narrated visualizations inside the Streamlit app.

#### Planned Insight Charts:

1. **Dialogue Volume by Character**
   - Bar chart showing total lines spoken by each character across all seasons
   - PyNarrative annotations: highlight who talks the most/least, surprising patterns

2. **Vocabulary Richness Comparison**
   - Unique words per character vs. total words spoken
   - Annotation: "Joey uses 40% fewer unique words than Ross, reflecting his simpler speech patterns"

3. **Character Dialogue Trends Across Seasons**
   - Line chart showing how each character's dialogue volume changes over 10 seasons
   - Annotations on key plot points (e.g., "Rachel's lines spike in Season 6 during her pregnancy arc")

4. **Word Embedding Clusters**
   - Scatter plot (t-SNE/PCA reduced) showing how character-specific words cluster
   - Annotations highlighting show-specific terms like "Pivot", "Unagi", "How you doin'"

5. **Sentiment Distribution by Character**
   - If sentiment analysis is added, show average sentiment per character
   - Annotations: "Phoebe's dialogue is the most positive, while Ross trends negative"

---

## 8. Frontend Design (Streamlit)

### App Layout — Multi-page Architecture:

#### 🏠 Main Page (`app.py`)
- App title: "The One With The AI"
- Brief description of the project
- Navigation via Streamlit's sidebar (auto-generated from `pages/` directory)

#### 🔍 Page 1: Search Quotes (`1_🔍_Search_Quotes.py`)
- Large search bar at the top
- "Search by meaning" tagline
- Results displayed as styled cards:
  - Character avatar (small)
  - Quote text (large)
  - Season & Episode info
  - Similarity score (subtle)
- Show top 5–10 results

#### 🎭 Page 2: Personality Quiz (`2_🎭_Personality_Quiz.py`)
- Display a scenario image using `st.image()`
- Text input field for user's reaction using `st.text_input()`
- "Analyze My Personality" button using `st.button()`
- Results section:
  - Winning character's avatar (large)
  - Character name and fun description
  - Progress bars for all 6 characters showing similarity %
  - Option to try another scenario

#### 📊 Page 3: Insights (`3_📊_Insights.py`)
- PyNarrative-powered annotated charts (see Feature C above)
- Each chart rendered via `st.altair_chart()`
- Organized as sections with headers and brief explanatory text

#### ℹ️ Page 4: About (`4_ℹ️_About.py`)
- How the project works (brief technical explanation)
- Technologies used
- Links to GitHub, Medium blog, LinkedIn
- Credits and data sources

---

## 9. Post-Project: Data Story with ipyvizzu-story

After the main project is complete, create an **animated data story** to showcase the project journey. This lives in `notebooks/04_data_story.ipynb` and can be exported as standalone HTML.

### Planned Story Slides:

| Slide | Content | Transition |
|---|---|---|
| 1 | "The Dataset" — Show raw data volume (episodes, lines, characters) | Fade in |
| 2 | "Cleaning the Data" — Show before/after word counts | Morph bar chart |
| 3 | "Training Word2Vec" — Show vocabulary growth over epochs | Animated line chart |
| 4 | "Character Fingerprints" — Compare character centroid vectors | Grouped bars morph to stacked |
| 5 | "Semantic Search in Action" — Show similarity score distributions | Scatter animation |
| 6 | "Results & Takeaways" — Final metrics and learnings | Summary slide |

### Usage:
- Export to **HTML** for embedding in Medium blog posts
- Include in the **GitHub repo** for interactive viewing
- Optional: Record as a GIF/video for README

---

## 10. Development Phases & Timeline

### Phase 1: Setup & Data (Day 1–2)
- [ ] Initialize project structure (folders, files)
- [ ] Download Friends script dataset from Kaggle
- [ ] Set up virtual environment and install dependencies
- [ ] Create `requirements.txt`
- [ ] Initial data exploration in `01_eda.ipynb`

### Phase 2: Preprocessing (Day 3–4)
- [ ] Build `preprocessing.py` with full cleaning pipeline
- [ ] Handle regex cleaning, tokenization, stopword filtering, lemmatization
- [ ] Validate cleaning with spot checks
- [ ] Document preprocessing in `02_preprocessing.ipynb`
- [ ] Save cleaned data to `data/processed/`

### Phase 3: Model Training (Day 5–6)
- [ ] Build `model.py` — Word2Vec training pipeline
- [ ] Train model with initial hyperparameters
- [ ] Run sanity checks (most_similar, similarity scores)
- [ ] Tune hyperparameters if quality is low
- [ ] Implement fallback to pre-trained embeddings if needed
- [ ] Document training process in `03_model_training.ipynb`
- [ ] Save model to `models/`

### Phase 4: Feature Development (Day 7–10)
- [ ] Build `search.py` — Semantic search logic
  - [ ] Sentence vectorization
  - [ ] Cosine similarity ranking
  - [ ] Top-K result retrieval
- [ ] Build `personality.py` — Personality matcher
  - [ ] Character centroid computation
  - [ ] Real-time matching logic
  - [ ] Result formatting
- [ ] Build `visualizations.py` — PyNarrative insight charts
  - [ ] Dialogue volume chart
  - [ ] Vocabulary richness chart
  - [ ] Trends across seasons chart
  - [ ] Word embedding cluster chart

### Phase 5: Frontend (Day 11–14)
- [ ] Set up Streamlit multi-page app structure
- [ ] Build Search Quotes page
- [ ] Build Personality Quiz page
- [ ] Build Insights page (PyNarrative integration)
- [ ] Build About page
- [ ] Gather and add image assets (character avatars, scenario images)
- [ ] Polish UI (styling, layout, responsiveness)
- [ ] Test all features end-to-end

### Phase 6: Deployment & Documentation (Day 15–17)
- [ ] Write comprehensive README.md
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Test deployed version

### Phase 7: Data Story & Blog (Day 18–20)
- [ ] Build ipyvizzu-story animated presentation in `04_data_story.ipynb`
- [ ] Export data story to HTML
- [ ] Draft Medium blog article in `docs/blog_draft.md`
- [ ] Publish blog and promote

---

## 11. Dependencies (requirements.txt)

```
# Core
pandas>=2.0
numpy>=1.24

# NLP
gensim>=4.3
spacy>=3.7
nltk>=3.8

# Machine Learning
scikit-learn>=1.3

# Visualization & Storytelling
altair>=5.0
pynarrative>=0.1
ipyvizzu-story>=0.9

# Web App
streamlit>=1.30

# Utilities
regex>=2023.0
```

**spaCy model (install separately):**
```
python -m spacy download en_core_web_sm
```

---

## 12. Key Design Decisions & Rationale

1. **Skip-gram over CBOW**: Better for small datasets — learns rare word representations more effectively.

2. **Selective stopword removal**: Unlike typical NLP tasks, personality matching needs sentiment words (`not`, `love`, `hate`). We keep them.

3. **Sentence vectors via averaging**: Simple but effective for this scale. More complex approaches (TF-IDF weighted averaging, Doc2Vec) can be explored if results are insufficient.

4. **PyNarrative for in-app, ipyvizzu-story for blog**: Each tool in its ideal context — PyNarrative's Altair output integrates natively with Streamlit; ipyvizzu-story's presentation format suits blog storytelling.

5. **Multi-page Streamlit app**: Cleaner UX than cramming everything into sidebars. Each feature gets its own dedicated page.

6. **Centroid-based personality matching**: Computationally lightweight, interpretable, and effective for a 6-class problem with substantial text per class.

---

## 13. Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Word2Vec vocabulary too small for meaningful embeddings | Search/matching quality poor | Fallback to pre-trained embeddings (GloVe/Google News); or use hybrid approach |
| Averaged sentence vectors lose meaning for short/sarcastic dialogue | Search results feel random | Experiment with TF-IDF weighted averaging; filter out very short lines (< 3 words) |
| Character centroids are too similar (all characters talk about similar topics) | Personality matcher always returns same character | Add TF-IDF weighting to emphasize *distinctive* words per character; increase vector dimensions |
| Kaggle dataset has formatting inconsistencies | Preprocessing breaks | Add robust error handling; inspect data thoroughly in EDA phase |
| PyNarrative or ipyvizzu-story API changes | Code breaks | Pin dependency versions in requirements.txt |

---

*Last updated: 18 February 2026*
*Author: Rishabh*
