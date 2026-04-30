import streamlit as st
import pandas as pd
import re
import nltk
import pickle
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="VibeCheck", page_icon="✌️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 620px !important; padding: 2rem 1.5rem 4rem !important; }

.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero-logo {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: white;
    letter-spacing: -3px;
    line-height: 1;
}
.hero-logo em { color: #FFE566; font-style: normal; }
.hero-sub {
    color: rgba(255,255,255,0.7);
    font-size: 12px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 8px;
    font-family: 'DM Sans', sans-serif;
}

.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.stat-card {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 1.2rem 0.5rem;
    text-align: center;
}
.stat-number {
    font-family: 'DM Sans', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: white;
    line-height: 1;
    display: block;
}
.stat-label {
    font-size: 11px;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
    display: block;
    font-family: 'DM Sans', sans-serif;
}

.section-label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}

.result-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}
.result-emoji { font-size: 60px; display: block; margin-bottom: 12px; }
.result-pos { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #059669; }
.result-neg { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #DC2626; }
.result-conf { font-size: 13px; color: #9CA3AF; margin-top: 6px; font-family: 'DM Sans', sans-serif; }
.bar-header { display: flex; justify-content: space-between; font-size: 12px; color: #9CA3AF; margin: 12px 0 4px; font-family: 'DM Sans', sans-serif; }
.bar-track { background: #F3F4F6; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-pos { height: 100%; border-radius: 999px; background: #059669; }
.bar-neg { height: 100%; border-radius: 999px; background: #DC2626; }

.hist-card {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.hist-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    font-size: 13px;
    font-family: 'DM Sans', sans-serif;
}
.hist-item:last-child { border-bottom: none; }
.hist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.hist-text { flex: 1; color: rgba(255,255,255,0.8); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge-pos { background: rgba(5,150,105,0.25); color: #6EE7B7; font-size: 11px; padding: 3px 10px; border-radius: 999px; flex-shrink: 0; }
.badge-neg { background: rgba(220,38,38,0.25); color: #FCA5A5; font-size: 11px; padding: 3px 10px; border-radius: 999px; flex-shrink: 0; }

/* Chip buttons */
div[data-testid="column"] div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    width: 100% !important;
}
div[data-testid="column"] div[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.25) !important;
    border-color: rgba(255,255,255,0.5) !important;
}

/* Textarea */
div[data-testid="stTextArea"] textarea {
    background: white !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 12px !important;
    color: #1A1A2E !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
}
div[data-testid="stTextArea"] label { display: none !important; }

/* Main button — target only the one NOT in a column */
.main-btn div[data-testid="stButton"] > button {
    background: white !important;
    color: #764ba2 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
    letter-spacing: 0.02em !important;
}
.main-btn div[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 25px rgba(0,0,0,0.2) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl") and os.path.exists("tfidf.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        return model, tfidf

    df = pd.read_csv("data/IMDB Dataset.csv")
    stop_words = set(stopwords.words('english'))

    def clean(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join([w for w in text.split() if w not in stop_words])

    df['clean'] = df['review'].apply(clean)
    X, y = df['clean'], df['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    model = LogisticRegression(max_iter=1000)
    model.fit(tfidf.fit_transform(X_train), y_train)

    with open("model.pkl", "wb") as f: pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
    return model, tfidf


def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join([w for w in text.split() if w not in stop_words])


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [('history', []), ('total', 0), ('pos', 0), ('neg', 0), ('text', '')]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">vibe<em>check</em></div>
    <div class="hero-sub">ML-powered sentiment analysis</div>
</div>
""", unsafe_allow_html=True)


# ── Stats ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <span class="stat-number">{st.session_state.total}</span>
        <span class="stat-label">analysed</span>
    </div>
    <div class="stat-card">
        <span class="stat-number" style="color:#6EE7B7">{st.session_state.pos}</span>
        <span class="stat-label">positive</span>
    </div>
    <div class="stat-card">
        <span class="stat-number" style="color:#FCA5A5">{st.session_state.neg}</span>
        <span class="stat-label">negative</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chips ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">try these</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
if c1.button("This film was absolutely fire 🔥", key="c1"):
    st.session_state.text = "This film was absolutely fire"
    st.rerun()
if c2.button("Total waste of 2 hours ngl", key="c2"):
    st.session_state.text = "Total waste of 2 hours"
    st.rerun()
if c3.button("Best thing I've watched all year", key="c3"):
    st.session_state.text = "Best thing I've watched all year"
    st.rerun()
if c4.button("Mid movie, not gonna lie", key="c4"):
    st.session_state.text = "Mid movie, not gonna lie"
    st.rerun()


# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label" style="margin-top:1rem">your review</p>', unsafe_allow_html=True)
review = st.text_area(
    "review",
    value=st.session_state.text,
    placeholder="Drop your review here... This film had me in my feels fr 🍿",
    height=100,
    label_visibility="collapsed"
)

st.markdown('<div class="main-btn">', unsafe_allow_html=True)
analyse = st.button("analyse the vibe ✌️")
st.markdown('</div>', unsafe_allow_html=True)


# ── Predict ────────────────────────────────────────────────────────────────────
if analyse and review.strip():
    model, tfidf = load_model()
    with st.spinner("checking vibes..."):
        vec = tfidf.transform([clean_text(review)])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

    pos_pct = round(proba[1] * 100)
    neg_pct = round(proba[0] * 100)
    is_pos = pred == 1
    conf = max(pos_pct, neg_pct)

    st.session_state.total += 1
    if is_pos: st.session_state.pos += 1
    else: st.session_state.neg += 1
    st.session_state.history.insert(0, {
        "text": review[:48] + ("..." if len(review) > 48 else ""),
        "is_pos": is_pos
    })
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

    emoji = "😊" if is_pos else "😞"
    label_class = "result-pos" if is_pos else "result-neg"
    label = "POSITIVE VIBES" if is_pos else "NEGATIVE VIBES"

    st.markdown(f"""
    <div class="result-card">
        <span class="result-emoji">{emoji}</span>
        <div class="{label_class}">{label}</div>
        <div class="result-conf">{conf}% confidence</div>
        <div class="bar-header"><span>positive</span><span>{pos_pct}%</span></div>
        <div class="bar-track"><div class="bar-pos" style="width:{pos_pct}%"></div></div>
        <div class="bar-header"><span>negative</span><span>{neg_pct}%</span></div>
        <div class="bar-track"><div class="bar-neg" style="width:{neg_pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.rerun()


# ── History ────────────────────────────────────────────────────────────────────
if st.session_state.history:
    items = ""
    for item in st.session_state.history:
        dot = "#059669" if item['is_pos'] else "#DC2626"
        badge = '<span class="badge-pos">positive</span>' if item['is_pos'] else '<span class="badge-neg">negative</span>'
        items += f"""
        <div class="hist-item">
            <div class="hist-dot" style="background:{dot}"></div>
            <div class="hist-text">{item['text']}</div>
            {badge}
        </div>"""
    st.markdown(f"""
    <div class="hist-card">
        <p class="section-label" style="margin-bottom:8px">recent checks</p>
        {items}
    </div>
    """, unsafe_allow_html=True)