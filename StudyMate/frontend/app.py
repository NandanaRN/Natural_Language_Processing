import streamlit as st
import re
import numpy as np
from collections import Counter
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")

# ---------------------------------------------------------------- styling --
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --primary: #6C5CE7;
  --primary-soft: #EDEAFC;
  --bg: #F8F7FF;
  --card: #FFFFFF;
  --text: #2D2A3A;
  --muted: #8A86A5;
  --strong: #27AE60;
  --practice: #F39C12;
  --weak: #E74C3C;
}

.stApp { background: linear-gradient(180deg, #F8F7FF 0%, #EFECFB 100%); }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1100px; }

/* Header */
.hero {
  background: linear-gradient(135deg, #6C5CE7 0%, #8E7CFF 55%, #A78BFA 100%);
  border-radius: 22px; padding: 2.2rem 2.5rem; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 12px 30px -12px rgba(108,92,231,.55);
  position: relative; overflow: hidden;
}
.hero::after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 230px; height: 230px;
  background: rgba(255,255,255,.12); border-radius: 50%;
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -.5px; }
.hero p { margin: .35rem 0 0; font-size: 1.02rem; opacity: .92; max-width: 760px; }
.hero .chip {
  display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  padding: .22rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: .4rem;
}

/* Cards */
.card {
  background: var(--card); border: 1px solid #ECE9FB; border-radius: 16px;
  padding: 1.1rem 1.3rem; margin-bottom: .9rem;
  box-shadow: 0 4px 14px -8px rgba(45,42,58,.12);
}
.card-title { font-weight: 700; font-size: 1rem; margin: 0 0 .15rem; }
.card-sub { color: var(--muted); font-size: .85rem; margin: 0; }

/* Topic pills */
.pill {
  display: inline-block; background: var(--primary-soft); color: var(--primary);
  border-radius: 999px; padding: .3rem .9rem; margin: .15rem .2rem;
  font-size: .82rem; font-weight: 600; border: 1px solid #DCD6F7;
}

/* Step badge */
.step {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--card); border: 1px solid #ECE9FB; border-radius: 12px;
  padding: .5rem 1rem; margin-bottom: 1rem; font-size: .82rem; font-weight: 600; color: var(--muted);
}
.step .num {
  width: 22px; height: 22px; border-radius: 50%; background: var(--primary); color: #fff;
  display: inline-flex; align-items: center; justify-content: center; font-size: .78rem; font-weight: 700;
}
.step.active { color: var(--primary); border-color: var(--primary); box-shadow: 0 2px 10px -4px rgba(108,92,231,.5); }

/* Badges */
.badge { display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 700; }
.badge-strong { background: #E8F8EF; color: var(--strong); border: 1px solid #BCEBD2; }
.badge-practice { background: #FEF5E7; color: var(--practice); border: 1px solid #F9E2B6; }
.badge-weak { background: #FDECEA; color: var(--weak); border: 1px solid #F6C6BF; }

/* Question card */
.quiz-card {
  background: var(--card); border: 1px solid #ECE9FB; border-radius: 18px;
  padding: 1.5rem 1.6rem; box-shadow: 0 8px 24px -12px rgba(108,92,231,.28);
}
.q-label { font-size: .78rem; font-weight: 700; letter-spacing: .6px; text-transform: uppercase; color: var(--primary); margin-bottom: .5rem; }
.q-text { font-size: 1.12rem; font-weight: 600; line-height: 1.5; margin: 0 0 1rem; }

/* Score */
.score-box { text-align: center; padding: 1rem; }
.score-num { font-size: 3rem; font-weight: 800; line-height: 1; }
.score-num.ok { color: var(--strong); } .score-num.mid { color: var(--practice); } .score-num.bad { color: var(--weak); }
.score-lbl { color: var(--muted); font-size: .85rem; font-weight: 600; }

/* Study links */
.link-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: .6rem; }
.btn-link {
  display: inline-block; background: var(--primary-soft); color: var(--primary);
  border: 1px solid #DCD6F7; border-radius: 10px; padding: .5rem 1rem;
  font-weight: 600; font-size: .85rem; text-decoration: none; transition: all .15s ease;
}
.btn-link:hover { background: var(--primary); color: #fff; }

/* Footer */
.footer { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 2.5rem; }

/* Streamlit widget polish */
.stButton > button {
  border-radius: 12px; font-weight: 600; border: none; padding: .55rem 1.2rem;
  transition: all .15s ease; box-shadow: 0 4px 12px -6px rgba(108,92,231,.4);
}
.stButton > button:hover { transform: translateY(-1px); }
.stTextArea textarea, .stTextInput input, .stSelectbox > div > div { border-radius: 12px !important; }
div[data-baseweb="select"] > div { border-radius: 12px !important; }
.stProgress > div > div > div { background: var(--primary); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SAMPLE_TEXT = """Natural Language Processing enables computers to understand human language.
Tokenization breaks text into tokens. TF-IDF represents terms using frequency and inverse document frequency.
Word embeddings represent words as dense vectors. Word2Vec learns representations from local context.
GloVe uses global word co-occurrence statistics. Transformers use self-attention to model relationships between tokens.
BERT produces contextual representations and is useful for classification and question answering."""

STOPWORDS = {
    "word", "words", "term", "terms", "text", "use", "uses", "using", "used", "represent",
    "represents", "representations", "language", "natural", "processing", "local",
    "context", "dense", "global", "attention", "statistics", "relationships", "tokens",
    "vectors", "human", "computers", "question", "answer", "answering", "classification",
    "frequency", "inverse", "understand", "enables", "learns", "produces", "useful",
    "model", "models", "breaks", "document", "works", "based", "made", "make", "many",
    "much", "one", "two", "also", "can", "like", "way", "ways", "help", "helps", "know",
}

# ---------------------------------------------------------------- helpers --
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_topics(sentences, top_n=10, min_len=4):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 1),
                          token_pattern=r"(?u)\b\w[\w-]*\w*\b")
    M = vec.fit_transform(sentences)
    terms = vec.get_feature_names_out()
    total = np.asarray(M.sum(axis=0)).ravel()
    topics = []
    for r in range(M.shape[0]):
        row = M.getrow(r)
        if row.nnz == 0:
            continue
        cols, vals = row.indices, row.data
        order = sorted(range(len(cols)), key=lambda k: (-vals[k], -total[cols[k]]))
        chosen = None
        for k in order:
            t = terms[cols[k]]
            if len(t) >= min_len and t not in STOPWORDS:
                chosen = t
                break
        if chosen is None or chosen in topics:
            continue
        topics.append(chosen)
        if len(topics) >= top_n:
            break
    return topics

def topic_sentences(topic, sentences):
    return [s for s in sentences if topic.lower() in s.lower()]

QUESTION_TEMPLATES = [
    ("Explain", "In your own words, explain what {topic} is and how it works."),
    ("How it works", "Describe step by step how {topic} works."),
    ("Why it matters", "Why is {topic} important and where is it used?"),
    ("Example", "Give a concrete example that shows {topic} in action."),
    ("Relation", "How does {topic} relate to the other concepts in this material?"),
]

def build_questions(topics, sentences, per_topic=3):
    questions = []
    qid = 0
    for topic in topics:
        relevant = topic_sentences(topic, sentences)
        if not relevant:
            continue
        for label, q in QUESTION_TEMPLATES[:per_topic]:
            questions.append({"id": qid, "topic": topic, "label": label,
                              "q": q.format(topic=topic)})
            qid += 1
    return questions

def build_practice(topics, sentences, per_topic=5):
    return build_questions(topics, sentences, per_topic=per_topic)

def grade(questions, answers, sentences):
    """Conceptual grading: an answer scores against the whole topic (definition,
    how it works, purpose) and keeps the best match, so understanding is rewarded
    over memorising exact wording."""
    model = load_model()
    cache = {}
    results = []
    for q in questions:
        text = (answers.get(q["id"]) or "").strip()
        n_words = len(text.split())
        if not text or n_words < 2:
            results.append({"id": q["id"], "topic": q["topic"], "q": q["q"],
                            "label": q["label"], "answer": text, "score": 0.0,
                            "answered": bool(text)})
            continue
        pool = topic_sentences(q["topic"], sentences)
        if not pool:
            pool = [q["q"]]
        pool = pool + [" ".join(pool)]
        a = model.encode(text, normalize_embeddings=True)
        best = 0.0
        for ref in pool:
            if ref not in cache:
                cache[ref] = model.encode(ref, normalize_embeddings=True)
            best = max(best, float(util.cos_sim(a, cache[ref])[0][0]) * 100)
        if n_words < 4:
            best *= 0.5
        results.append({"id": q["id"], "topic": q["topic"], "q": q["q"],
                        "label": q["label"], "answer": text, "score": best,
                        "answered": True})
    return results

def status(score):
    if score >= 65:
        return "strong", "Strong"
    if score >= 45:
        return "practice", "Needs practice"
    return "weak", "Weak"

def teaching_block(topic, sentences):
    refs = topic_sentences(topic, sentences)
    explanation = " ".join(dict.fromkeys(refs))
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b", explanation.lower())
    keep = [w for w in words if w not in STOPWORDS]
    key = [w for w, _ in Counter(keep).most_common(5)]
    wiki = "https://en.wikipedia.org/wiki/" + quote(topic.replace(" ", "_"))
    search = "https://duckduckgo.com/?q=" + quote(topic + " explained tutorial")
    return explanation, key, wiki, search

def hero():
    st.markdown("""
    <div class="hero">
      <h1>StudyMate</h1>
      <p>Upload your study material, pick the topics you want to master, and take a quiz.
      StudyMate grades your answers for understanding, not memory, finds your weak spots,
      teaches you what you missed and points you to the right material to revise.</p>
      <div>
        <span class="chip">Topic Extraction</span>
        <span class="chip">Smart Quiz</span>
        <span class="chip">Weakness Analysis</span>
        <span class="chip">Guided Revision</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def step_badge(num, label, active=False):
    cls = "step active" if active else "step"
    st.markdown(f'<div class="{cls}"><span class="num">{num}</span>{label}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- session --
DEFAULTS = {
    "topics": [], "sentences": [], "selected": [], "questions": [],
    "q_index": 0, "answers": {}, "submitted": False, "results": [], "mode": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------- page --
hero()

with st.sidebar:
    st.markdown("## StudyMate")
    st.caption("AI learning assistant that tests your understanding, finds weak areas and guides your revision.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
    1. Upload or use sample material
    2. Extract key topics
    3. Choose what to be tested on
    4. Answer every question
    5. See weak areas, learn, retake
    """)
    st.divider()
    st.caption("Grading is based on conceptual understanding (definition and how it works), not memorisation.")

# ------------------------------------------------- STEP 1: upload & analyze
step_badge(1, "Upload study material", active=not st.session_state["topics"])
uploaded = st.file_uploader("Upload study material (TXT)", type=["txt"])
use_sample = st.checkbox("Use built-in sample material instead", value=not uploaded)

text = ""
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")
elif use_sample:
    text = SAMPLE_TEXT

if text and st.button("Analyze Study Material", type="primary"):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
    topics = extract_topics(sentences) if sentences else []
    st.session_state.update({
        "topics": list(topics), "sentences": sentences,
        "questions": [], "q_index": 0, "answers": {},
        "submitted": False, "results": [], "selected": [], "mode": "",
    })
    st.toast("Material analyzed. Pick your topics below.", icon=None)

if st.session_state["topics"]:
    st.markdown('<div class="card"><p class="card-title">Topics extracted</p>'
                '<p class="card-sub">These are the most important concepts found in your material.</p><div style="margin-top:.6rem">'
                + "".join(f'<span class="pill">{t}</span>' for t in st.session_state["topics"])
                + "</div></div>", unsafe_allow_html=True)

# ------------------------------------------- STEP 2: choose topics -> quiz
if st.session_state["topics"] and not st.session_state["questions"] and not st.session_state["submitted"]:
    step_badge(2, "Choose topics and start the quiz", active=True)
    st.session_state["selected"] = st.multiselect(
        "Which topics should we test you on?",
        options=st.session_state["topics"],
        default=st.session_state["selected"] or st.session_state["topics"],
    )
    if st.button("Start Quiz", type="primary", disabled=not st.session_state["selected"]):
        questions = build_questions(st.session_state["selected"], st.session_state["sentences"])
        if questions:
            st.session_state.update({"questions": questions, "q_index": 0,
                                     "answers": {}, "submitted": False,
                                     "results": [], "mode": "quiz"})
            st.rerun()
        else:
            st.warning("No questions could be generated. Try picking different topics.")

# ------------------------------------------------------- STEP 3: the quiz
if st.session_state["questions"] and not st.session_state["submitted"]:
    qs = st.session_state["questions"]
    idx = st.session_state["q_index"]
    q = qs[idx]
    qkey = f"ans_{q['id']}"

    if st.session_state["mode"] == "practice":
        step_badge(3, "Focused practice on your weak topics", active=True)
    else:
        step_badge(3, f"Answer every question - {idx + 1} of {len(qs)}", active=True)

    st.progress((idx + 1) / len(qs), text=f"Question {idx + 1} of {len(qs)}")
    st.markdown(
        f'<div class="quiz-card">'
        f'<div class="q-label">{q["label"]} - {q["topic"]}</div>'
        f'<p class="q-text">{q["q"]}</p></div>',
        unsafe_allow_html=True,
    )

    st.text_area("Your answer", key=qkey, height=130,
                 placeholder="Write your answer here, in your own words.", label_visibility="collapsed")

    prev_col, next_col = st.columns([1, 1])
    if idx > 0 and prev_col.button("Previous", use_container_width=True):
        st.session_state["answers"][q["id"]] = (st.session_state.get(qkey) or "").strip()
        st.session_state["q_index"] = idx - 1
        st.rerun()
    if idx < len(qs) - 1:
        if next_col.button("Save and Continue", type="primary", use_container_width=True):
            st.session_state["answers"][q["id"]] = (st.session_state.get(qkey) or "").strip()
            st.session_state["q_index"] = idx + 1
            st.rerun()
    else:
        if next_col.button("Submit Answers", type="primary", use_container_width=True):
            st.session_state["answers"][q["id"]] = (st.session_state.get(qkey) or "").strip()
            st.session_state["results"] = grade(qs, st.session_state["answers"],
                                                st.session_state["sentences"])
            st.session_state["submitted"] = True
            st.rerun()

# --------------------------------------------- STEP 4: analysis + study plan
if st.session_state["submitted"] and st.session_state["results"]:
    results = st.session_state["results"]
    scores = [r["score"] for r in results]
    overall = float(np.mean(scores))

    if st.session_state["mode"] == "practice":
        step_badge(4, "Practice complete - review your progress", active=True)
    else:
        step_badge(4, "Results and personalised study plan", active=True)

    overall_cls = "ok" if overall >= 65 else ("mid" if overall >= 45 else "bad")
    overall_lbl = ("Strong understanding" if overall >= 65 else
                   "Good base, needs a little more practice" if overall >= 45 else
                   "Needs focused revision")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div class="card score-box">'
            f'<div class="score-num {overall_cls}">{overall:.0f}<span style="font-size:1.4rem">%</span></div>'
            f'<div class="score-lbl">Overall understanding</div>'
            f'<div style="margin-top:.4rem;font-weight:600">{overall_lbl}</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><p class="card-title">Topic-wise breakdown</p>'
                    '<p class="card-sub">Your score per topic - green is solid, red needs revision.</p></div>',
                    unsafe_allow_html=True)

    per_topic = {}
    for r in results:
        per_topic.setdefault(r["topic"], []).append(r["score"])
    topics_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:.45rem 0;'
        f'border-bottom:1px solid #F1EEFC">'
        f'<span style="font-weight:600">{t}</span>'
        f'<span style="display:flex;gap:.6rem;align-items:center"><b style="min-width:3rem;text-align:right">{np.mean(v):.0f}%</b>'
        f'<span class="badge badge-{status(np.mean(v))[0]}">{status(np.mean(v))[1]}</span></span></div>'
        for t, v in per_topic.items()
    )
    st.markdown(
        f'<div class="card" style="padding:1rem 1.4rem"><div style="display:flex;flex-direction:column">'
        + topics_rows + "</div></div>", unsafe_allow_html=True)

    weak = {t: np.mean(v) for t, v in per_topic.items() if np.mean(v) < 45}
    practice = {t: np.mean(v) for t, v in per_topic.items() if 45 <= np.mean(v) < 65}
    strong = {t: np.mean(v) for t, v in per_topic.items() if np.mean(v) >= 65}

    if weak:
        st.markdown("### Study these concepts first")
        st.markdown(
            f'<div class="card" style="border-color:#F6C6BF;background:#FFF7F6">'
            f'<p class="card-title" style="color:#E74C3C">Your weakest topics - learn these, then retake the quiz</p>'
            f'<p class="card-sub">For each topic below you get a short lesson from your material, key points to remember,'
            f' external study links, and a focused practice set.</p></div>',
            unsafe_allow_html=True)
        for t, s in sorted(weak.items(), key=lambda kv: kv[1]):
            explanation, key, wiki, search = teaching_block(t, st.session_state["sentences"])
            key_pills = "".join(f'<span class="pill">{k}</span>' for k in key)
            st.markdown(
                f'<div class="card">'
                f'<p class="card-title">{t} <span class="badge badge-weak">Score {s:.0f}%</span></p>'
                f'<p class="card-sub" style="margin-bottom:.4rem">What you should understand:</p>'
                f'<div style="background:#FAF9FF;border:1px solid #ECE9FB;border-radius:10px;padding:.8rem 1rem;'
                f'line-height:1.7;color:#4A4663;font-size:.95rem">{explanation}</div>'
                f'<p class="card-sub" style="margin:.7rem 0 .3rem">Key points to remember:</p>'
                f'<div>{key_pills}</div>'
                f'<div class="link-row">'
                f'<a class="btn-link" href="{wiki}" target="_blank">Read more on Wikipedia</a>'
                f'<a class="btn-link" href="{search}" target="_blank">Search for tutorials</a>'
                f'</div></div>', unsafe_allow_html=True)
        if st.button("Practice your weak topics - more questions", type="primary",
                     use_container_width=True):
            practice_qs = build_practice(list(weak), st.session_state["sentences"], per_topic=5)
            st.session_state.update({"questions": practice_qs, "q_index": 0,
                                     "answers": {}, "submitted": False,
                                     "results": [], "mode": "practice"})
            st.rerun()

    if practice:
        st.markdown("### Sharpen these topics")
        for t, s in sorted(practice.items(), key=lambda kv: kv[1]):
            explanation, key, wiki, search = teaching_block(t, st.session_state["sentences"])
            key_pills = "".join(f'<span class="pill">{k}</span>' for k in key)
            st.markdown(
                f'<div class="card" style="border-color:#F9E2B6;background:#FFFDF6">'
                f'<p class="card-title">{t} <span class="badge badge-practice">Score {s:.0f}%</span></p>'
                f'<p class="card-sub">Good start, but worth another pass. Re-read the material below and retry.</p>'
                f'<div style="background:#fff;border:1px solid #F9E2B6;border-radius:10px;padding:.8rem 1rem;'
                f'line-height:1.7;color:#4A4663;font-size:.95rem">{explanation}</div>'
                f'<div>{key_pills}</div>'
                f'<div class="link-row">'
                f'<a class="btn-link" href="{wiki}" target="_blank">Read more on Wikipedia</a>'
                f'<a class="btn-link" href="{search}" target="_blank">Search for tutorials</a>'
                f'</div></div>', unsafe_allow_html=True)

    if strong:
        st.markdown("### You've got these")
        st.markdown('<div class="card">' + "".join(
            f'<span class="pill" style="background:#E8F8EF;color:#27AE60;border-color:#BCEBD2">{t} - {np.mean(v):.0f}%</span>'
            for t, v in strong.items()) + "</div>", unsafe_allow_html=True)

    with st.expander("View every answer and score"):
        for r in results:
            st.markdown(f"**{r['label']} - {r['topic']}** - *{r['score']:.0f}%*")
            st.caption(r["q"])
            st.write(f"Your answer: {r['answer'] if r['answered'] else '(skipped)'}")

    b1, b2 = st.columns(2)
    if b1.button("Retake Quiz", use_container_width=True):
        st.session_state.update({"answers": {}, "q_index": 0,
                                 "submitted": False, "results": []})
        st.rerun()
    if b2.button("Start Over with New Material", use_container_width=True):
        st.session_state.update(DEFAULTS)
        st.rerun()

elif not st.session_state["topics"]:
    st.info("Upload your study material (or use the sample) and click Analyze Study Material to begin.")

st.markdown('<div class="footer">StudyMate - NLP mini-project demo - semantic answer evaluation with sentence embeddings</div>',
            unsafe_allow_html=True)