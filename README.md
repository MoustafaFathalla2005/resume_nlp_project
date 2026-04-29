# Resume NLP System 📄

A beginner-friendly NLP project that classifies resumes, matches them to job descriptions, and generates job postings — **built step by step, no black-box models**.

---

## What the project does

| Page | Feature |
|------|---------|
| 🏠 Home | Pipeline overview and dataset statistics |
| 🤖 Classifier | Paste a resume → predict its job category |
| 🔍 JD Matcher | Paste a Job Description → find the most similar resumes |
| 📝 JD Generator | Paste a resume → get a matching Job Description (rule-based or Claude AI) |
| 📊 EDA | Explore the dataset with charts |

---

## How the pipeline works (no magic)

```
Raw Resume Text
      │
      ▼
 1. ResumeCleaner        → remove emails, phones, URLs, special chars
      │
      ▼
 2. ResumePreprocessor   → word_tokenize → remove stop words → lemmatise
      │
      ▼
 3. ResumeClassifier     → TF-IDF weights → one centroid per category
      │                    → cosine similarity → predicted category
      ▼
 4. ResumeMatcher        → same TF-IDF → cosine similarity vs. JD
      │
      ▼
 5. JobDescriptionGenerator → rule-based template  OR  Claude API
```

### Concepts used (from the course notebooks)

- **Tokenisation** — `word_tokenize()` from NLTK (Week 9)
- **Stop-word removal** — `stopwords.words('english')` (Week 9)
- **Lemmatisation** — `WordNetLemmatizer` (Week 9)
- **TF-IDF** — `TfidfVectorizer` from sklearn
- **Cosine similarity** — `cosine_similarity` from sklearn
- **Bag of N-Grams** — `CountVectorizer` (nlp_P_preprocessing notebook)
- **NER / POS tagging** concepts — Week 11 (used for understanding)

---

## Project structure

```
resume_nlp_project/
├── app.py              ← Streamlit app (5 pages)
├── cleaner.py          ← ResumeCleaner class
├── preprocessor.py     ← ResumePreprocessor class (NLTK)
├── vectorizer.py       ← BagOfNGrams class (CountVectorizer wrapper)
├── classifier.py       ← ResumeClassifier (TF-IDF centroid, no sklearn model)
├── matcher.py          ← ResumeMatcher (cosine similarity)
├── jd_generator.py     ← JobDescriptionGenerator (rule-based + Claude AI)
├── ResumeDataSet.csv   ← dataset (place here before running)
└── README.md
```

---

## Setup and run

```bash
# 1. Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn nltk

# 2. Place the dataset next to app.py
#    (file: ResumeDataSet.csv  columns: Category, Resume)

# 3. Run the app
streamlit run app.py
```

The app automatically downloads the required NLTK data (`punkt`, `stopwords`, `wordnet`) on first run.

---

## Dataset

The project uses **ResumeDataSet.csv** (962 resumes, 25 categories).

| Column | Description |
|--------|-------------|
| `Category` | Job category (e.g. Data Science, Java Developer, HR) |
| `Resume` | Raw resume text |

If the CSV is not found, a small demo dataset is loaded automatically so the app still works.

---

## Classifier — how it works without sklearn models

Instead of LogisticRegression or SVM we use a **Centroid Classifier**:

1. Fit TF-IDF on all resumes → sparse matrix `(n_docs, n_features)`
2. For each category, average all its TF-IDF row vectors → one **centroid** vector per category
3. For a new resume, transform it to TF-IDF and compute **cosine similarity** to every centroid
4. The category with the highest similarity wins

This is simple, transparent, and works well on text data.

---

## JD Generator — AI mode

When "Use Claude AI" is toggled on, the app sends the resume to the **Claude API** and returns a unique, non-repetitive job description. No API key needs to be configured separately — the Anthropic SDK handles it automatically inside the Streamlit environment.

---

## Team

Built as a group project for the NLP course.
