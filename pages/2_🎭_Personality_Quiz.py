"""Personality Quiz — Find out which Friend you are!"""
import streamlit as st
import os

st.set_page_config(page_title="Personality Quiz", page_icon="🎭", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .quiz-scenario {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .scenario-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .scenario-text {
        font-size: 1.15rem;
        color: #e0e0e0;
        font-weight: 500;
    }
    .winner-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 2px solid #F5A623;
    }
    .winner-name {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F5A623;
    }
    .winner-pct {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #F5A623, #E8575A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .bar-container {
        display: flex;
        align-items: center;
        margin: 6px 0;
    }
    .bar-label {
        width: 90px;
        text-align: right;
        padding-right: 10px;
        font-size: 0.9rem;
        color: #ccc;
    }
    .bar-fill {
        height: 22px;
        border-radius: 11px;
        transition: width 0.5s ease;
    }
    .bar-pct {
        padding-left: 8px;
        font-size: 0.85rem;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Character colors
CHAR_COLORS = {
    "Rachel": "#E8575A",
    "Ross": "#5B8DB8",
    "Chandler": "#F5A623",
    "Monica": "#7B68EE",
    "Joey": "#50C878",
    "Phoebe": "#FF69B4",
}

# Character images
CHAR_IMAGES = {}
img_dir = "assets/characters"
for name in CHAR_COLORS:
    img_path = os.path.join(img_dir, f"{name.lower()}.jpg")
    if os.path.exists(img_path):
        CHAR_IMAGES[name] = img_path


# Scenarios for the quiz
SCENARIOS = [
    {
        "emoji": "🧹",
        "title": "Your apartment is a mess",
        "prompt": "You walk into your apartment and it's a total disaster. How do you react?",
    },
    {
        "emoji": "💔",
        "title": "Your partner wants a break",
        "prompt": "Your significant other says they need some space. What do you say?",
    },
    {
        "emoji": "🍕",
        "title": "Pizza night",
        "prompt": "Your friends ordered pizza and there's only one slice left. What do you do?",
    },
    {
        "emoji": "🎸",
        "title": "Open mic night",
        "prompt": "It's open mic night at the coffee shop. You have to perform something. What would you do?",
    },
    {
        "emoji": "💼",
        "title": "Job opportunity",
        "prompt": "You got offered your dream job but it's in another city. How do you decide?",
    },
]


@st.cache_resource
def load_matcher():
    """Load the personality matcher (cached)."""
    from src.personality import build_personality_matcher
    return build_personality_matcher()


st.title("🎭 Personality Quiz")
st.markdown("React to **5 scenarios** and find out which Friend you're most like!")
st.markdown("---")

# Load matcher
with st.spinner("Loading personality engine..."):
    matcher = load_matcher()

# ── Quiz Form ─────────────────────────────────────────────────────────────────
answers = []
with st.form("quiz_form"):
    for i, scenario in enumerate(SCENARIOS):
        st.markdown(
            f"""
            <div class="quiz-scenario">
                <div class="scenario-label">{scenario['emoji']} Scenario {i+1} of {len(SCENARIOS)}</div>
                <div class="scenario-text">{scenario['prompt']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        answer = st.text_area(
            f"Your reaction:",
            placeholder="Type your response here...",
            height=80,
            key=f"scenario_{i}",
            label_visibility="collapsed",
        )
        answers.append(answer)
        st.markdown("")

    submitted = st.form_submit_button(
        "🎯 Reveal My Match!",
        use_container_width=True,
        type="primary",
    )

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    filled = [a for a in answers if a.strip()]
    if len(filled) < 2:
        st.warning("Please fill in at least 2 scenarios for an accurate match!")
    else:
        with st.spinner("Analyzing your personality..."):
            results = matcher.match_multiple(filled)

        winner = results[0]
        st.markdown("---")

        # Winner card
        col_img, col_result = st.columns([1, 2])

        with col_img:
            if winner["character"] in CHAR_IMAGES:
                st.image(CHAR_IMAGES[winner["character"]], width=250)

        with col_result:
            st.markdown(
                f"""
                <div class="winner-card">
                    <div style="font-size: 1rem; color: #888;">You are most like...</div>
                    <div class="winner-name">{winner['character']}</div>
                    <div class="winner-pct">{winner['percentage']:.1f}%</div>
                    <div style="font-size: 0.9rem; color: #aaa;">personality match</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # All character bars
        st.markdown("")
        st.markdown("#### Full Breakdown")
        for r in results:
            color = CHAR_COLORS.get(r["character"], "#888")
            width = max(r["percentage"], 2)
            st.markdown(
                f"""
                <div class="bar-container">
                    <div class="bar-label">{r['character']}</div>
                    <div class="bar-fill" style="width: {width}%; background: {color};"></div>
                    <div class="bar-pct">{r['percentage']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.info(
            "💡 **How it works:** Your responses are analyzed using Sentence-BERT "
            "(contextual embeddings), Word2Vec (Friends-specific patterns), and "
            "topic keyword matching — blended together for the final score.",
            icon="🧠",
        )
