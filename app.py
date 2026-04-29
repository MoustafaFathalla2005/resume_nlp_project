"""
app.py — Resume NLP System (Streamlit)
=======================================
Pages:
    1. Home        — pipeline overview and dataset stats
    2. Classifier  — paste a resume, get a predicted job category
    3. JD Matcher  — paste a Job Description, get the top matching resumes
    4. JD Generator— paste a resume, generate a unique Job Description
    5. EDA         — explore the dataset with charts

Run:
    streamlit run app.py

Requirements:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn nltk
"""

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from cleaner      import ResumeCleaner
from preprocessor import ResumePreprocessor
from classifier   import ResumeClassifier
from matcher      import ResumeMatcher
from jd_generator import JobDescriptionGenerator

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Resume NLP",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #6c63ff; }
    .metric-label { font-size: 0.82rem; color: grey; margin-top: 4px; }

    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        background: #6c63ff;
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 3px;
    }

    .jd-block {
        background: var(--secondary-background-color);
        border-left: 4px solid #6c63ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        font-family: monospace;
        font-size: 0.86rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load data and build pipeline ─────────────────────────────────────────── #

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Build every component of the NLP pipeline from scratch.

    Steps:
        1. Load the CSV dataset
        2. Clean raw text with ResumeCleaner
        3. Tokenise + lemmatise with ResumePreprocessor
        4. Fit ResumeClassifier  (TF-IDF centroid classifier)
        5. Fit ResumeMatcher     (TF-IDF cosine similarity)

    Returns
    -------
    dict with keys: cleaner, preprocessor, classifier, matcher, df
    """
    df = _load_dataframe()

    cleaner = ResumeCleaner()
    preproc = ResumePreprocessor()

    # clean and process every resume
    df["cleaned"]           = cleaner.clean_series(df["Resume"])
    df["processed_resume"]  = preproc.process_series(df["cleaned"])

    # drop rows where processing left nothing useful
    df = df[df["processed_resume"].str.len() > 10].reset_index(drop=True)

    # fit classifier and matcher on the processed text
    classifier = ResumeClassifier(max_features=10_000)
    classifier.fit(df, text_col="processed_resume", label_col="Category")

    matcher = ResumeMatcher(max_features=10_000)
    matcher.fit(df, text_col="processed_resume")

    return {
        "cleaner"    : cleaner,
        "preprocessor": preproc,
        "classifier" : classifier,
        "matcher"    : matcher,
        "df"         : df,
    }


def _load_dataframe():
    """Try several paths for ResumeDataSet.csv; return a DataFrame."""
    for path in ["ResumeDataSet.csv", "../ResumeDataSet.csv", "data/ResumeDataSet.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if {"Category", "Resume"}.issubset(df.columns):
                return df.dropna(subset=["Category", "Resume"]).reset_index(drop=True)

    # small fallback demo dataset so the app can run without a CSV
    return pd.DataFrame({
        "Category": (["Data Science"] * 3 + ["Java Developer"] * 3 +
                     ["Web Designing"] * 3 + ["Python Developer"] * 3 +
                     ["DevOps Engineer"] * 3 + ["Testing"] * 2),
        "Resume": [
            "Data scientist 5 years Python machine learning TensorFlow deep learning NLP pandas sklearn",
            "ML engineer PyTorch deep learning neural networks NLP AWS MLOps model deployment",
            "Data analyst SQL Tableau Power BI Python statistics A/B testing dashboards",
            "Java developer 6 years Spring Boot microservices REST Kafka Docker Kubernetes PostgreSQL",
            "Senior Java Spring Cloud distributed systems MySQL Redis Docker CI/CD Jenkins",
            "Backend Java Spring MVC Hibernate JPA Oracle Gradle JUnit integration testing",
            "Frontend React JavaScript TypeScript HTML CSS Redux responsive webpack",
            "Full stack React Node Express MongoDB REST GraphQL Docker AWS deployment",
            "UI UX designer Figma Adobe XD user research prototyping Vue CSS",
            "Python Django REST API PostgreSQL Redis Celery Docker AWS authentication",
            "Python Flask FastAPI async SQLAlchemy PostgreSQL Redis Docker",
            "Backend Python microservices Kafka MongoDB Docker Kubernetes CI/CD",
            "DevOps Docker Kubernetes Jenkins Terraform Ansible AWS EKS Prometheus Grafana",
            "Cloud AWS Azure GCP Terraform CloudFormation Lambda RDS IAM security",
            "Platform Kubernetes Helm Istio GitOps ArgoCD infrastructure Linux bash",
            "QA Selenium Python pytest Cucumber JIRA Postman regression API testing",
            "SDET Java Selenium BDD Gherkin JMeter performance testing CI/CD",
        ],
    })


def _classify(text, pipeline):
    """Run the full pipeline on a resume string and return a result dict."""
    cleaned   = pipeline["cleaner"].clean(text)
    processed = pipeline["preprocessor"].process(cleaned)
    top_k     = pipeline["classifier"].predict_topk(processed, k=5)
    return {
        "category" : top_k[0]["category"],
        "score"    : top_k[0]["score"],
        "top_k"    : top_k,
        "cleaned"  : cleaned,
        "processed": processed,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("## 📄 Resume NLP")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🤖 Classifier", "🔍 JD Matcher", "📝 JD Generator", "📊 EDA"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Stack:** NLTK · TF-IDF · Cosine · Streamlit")

# ── Load pipeline ─────────────────────────────────────────────────────────── #
with st.spinner("Building pipeline…"):
    pipeline = load_pipeline()

df          = pipeline["df"]
STOP_WORDS  = pipeline["preprocessor"]._stopwords


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📄 Resume NLP System")
    st.markdown(
        "An NLP pipeline that **classifies** resumes, **matches** them to job "
        "descriptions, and **generates** JDs — built step-by-step with NLTK and TF-IDF."
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, val, label in [
        (c1, f"{len(df):,}", "Resumes loaded"),
        (c2, str(df["Category"].nunique()), "Job categories"),
        (c3, "TF-IDF + Cosine", "Model"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pipeline steps")
    st.markdown("""
| Step | File | What it does |
|------|------|--------------|
| 1 | `cleaner.py` | Remove emails, phones, URLs, special characters |
| 2 | `preprocessor.py` | word_tokenize → stop-word removal → lemmatise (NLTK) |
| 3 | `vectorizer.py` | Bag-of-N-grams with CountVectorizer |
| 4 | `classifier.py` | TF-IDF centroids + cosine similarity (no sklearn model) |
| 5 | `matcher.py` | Rank resumes by cosine similarity to a JD |
| 6 | `jd_generator.py` | Rule-based or Claude-AI job description |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Classifier
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Classifier":
    st.header("Resume Classifier")
    st.markdown("Paste a resume — the system predicts its job category using cosine similarity to TF-IDF category centroids.")

    text_in = st.text_area("Resume text", height=220,
                            placeholder="e.g. Python developer with 5 years in machine learning, TensorFlow, scikit-learn...")

    if st.button("Classify", type="primary", use_container_width=True):
        if len(text_in.strip()) < 20:
            st.warning("Please enter at least 20 characters.")
        else:
            with st.spinner("Classifying…"):
                result = _classify(text_in, pipeline)

            st.markdown("---")
            col_a, col_b = st.columns([2, 3])

            with col_a:
                st.markdown("#### Predicted category")
                st.markdown(f'<span class="badge">🏷 {result["category"]}</span>',
                            unsafe_allow_html=True)
                st.markdown(f"**Similarity score:** `{result['score']*100:.1f}%`")
                st.progress(min(result["score"], 1.0))

            with col_b:
                st.markdown("#### Top 5 matches")
                for item in result["top_k"]:
                    pct = item["score"] * 100
                    st.markdown(f"{item['category']} — **{pct:.1f}%**")
                    st.progress(min(item["score"], 1.0))

            with st.expander("Processing details"):
                t1, t2 = st.tabs(["Cleaned text", "Processed text"])
                t1.code(result["cleaned"][:600], language="text")
                t2.code(result["processed"][:600], language="text")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — JD Matcher
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 JD Matcher":
    st.header("Job Description Matcher")
    st.markdown("Paste a Job Description — the system finds the most similar resumes.")

    jd_in = st.text_area("Job Description", height=160,
                           placeholder="e.g. Looking for a Python data scientist with TensorFlow and NLP experience...")

    col1, col2 = st.columns(2)
    top_n      = col1.slider("Results to show", 3, 15, 5)
    cat_filter = col2.selectbox(
        "Filter by category (optional)",
        ["All"] + sorted(df["Category"].unique().tolist()),
    )

    if st.button("Find matches", type="primary", use_container_width=True):
        if len(jd_in.strip()) < 10:
            st.warning("Please enter a longer job description.")
        else:
            with st.spinner("Searching…"):
                cleaner = pipeline["cleaner"]
                preproc = pipeline["preprocessor"]
                matcher = pipeline["matcher"]

                # clean and process the JD exactly like the resumes
                processed_jd = preproc.process(cleaner.clean(jd_in))
                cat          = None if cat_filter == "All" else cat_filter
                results      = matcher.match(processed_jd, top_n=top_n, category_filter=cat)

            st.markdown(f"### Top {top_n} matches")
            for _, row in results.iterrows():
                with st.expander(
                    f"#{int(row['rank'])}  {row['Category']}  —  {row['similarity_pct']}% similar",
                    expanded=(row["rank"] == 1),
                ):
                    st.progress(min(row["similarity_pct"] / 100, 1.0))
                    st.text(row["resume_snippet"])


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — JD Generator
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 JD Generator":
    st.header("Job Description Generator")
    st.markdown("Paste a resume — the system builds a matching Job Description.")

    col1, col2 = st.columns([3, 1])
    resume_in  = col1.text_area("Resume text", height=200, placeholder="Paste resume here…")
    use_ai     = col2.toggle("Use Claude AI", value=False)
    if use_ai:
        col2.info("Calls Claude API. Needs internet access and API key set in the environment.")

    if st.button("Generate JD", type="primary", use_container_width=True):
        if len(resume_in.strip()) < 20:
            st.warning("Please enter a longer resume.")
        else:
            with st.spinner("Generating…"):
                clf_result = _classify(resume_in, pipeline)
                gen        = JobDescriptionGenerator(use_ai=use_ai)
                jd_out     = gen.generate_from_resume(resume_in, category=clf_result["category"])

            st.success(f"Generated for category: **{clf_result['category']}**")
            st.markdown(f'<div class="jd-block">{jd_out}</div>', unsafe_allow_html=True)
            st.download_button(
                "Download JD (.txt)",
                data      = jd_out,
                file_name = f"JD_{clf_result['category'].replace(' ', '_')}.txt",
                mime      = "text/plain",
            )

            # show extracted skills as badges
            skills = gen.extract_skills(resume_in)
            if skills:
                with st.expander("Extracted skills"):
                    for s in skills:
                        st.markdown(f'<span class="badge">{s}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — EDA Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.header("EDA Dashboard")

    tab1, tab2, tab3 = st.tabs(["Category distribution", "Text length", "Top keywords"])

    with tab1:
        dist = df["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(x=dist.values, y=dist.index, palette="viridis", ax=ax)
        ax.set_title("Resume Category Distribution", fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        c1.metric("Total resumes", f"{len(df):,}")
        c2.metric("Categories",    df["Category"].nunique())

    with tab2:
        df["char_len"]   = df["Resume"].fillna("").apply(len)
        df["word_count"] = df["Resume"].fillna("").apply(lambda t: len(t.split()))

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(df["char_len"],   bins=40, color="#6c63ff", alpha=0.85, edgecolor="none")
        axes[0].set_title("Character length distribution", fontweight="bold")
        axes[1].hist(df["word_count"], bins=40, color="#ff9800", alpha=0.85, edgecolor="none")
        axes[1].set_title("Word count distribution", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Statistics")
        st.dataframe(df[["char_len", "word_count"]].describe().round(0))

    with tab3:
        cat    = st.selectbox("Select category", sorted(df["Category"].unique()))
        tokens = " ".join(df[df["Category"] == cat]["Resume"].fillna("")).lower().split()
        tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 3]
        top20  = Counter(tokens).most_common(20)

        if top20:
            words, freqs = zip(*top20)
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.bar(words, freqs, color="#6c63ff", edgecolor="none")
            ax.set_title(f"Top 20 keywords — {cat}", fontweight="bold")
            ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data for this category.")
