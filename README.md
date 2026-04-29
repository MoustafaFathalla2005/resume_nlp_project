# 📄 Resume NLP System

<p align="center">
  <img src="banner.png"/>
</p>

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Cosine%20Similarity-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

> 🚀 An end-to-end NLP system for resume classification, job matching, and job description generation using interpretable techniques.

---

## 🎬 Demo

<p align="center">
  <img src="demo.gif" width="900"/>
</p>

---

## 🚀 Overview

**Resume NLP System** is a complete Natural Language Processing application that transforms raw resumes into structured insights.

### 💡 What it does:

* 📂 Classifies resumes into job categories
* 🔍 Matches resumes with job descriptions
* 📝 Generates job descriptions from resumes

---

## ✨ Features

* 🤖 Resume Classification (TF-IDF + Centroid)
* 🔍 Smart Job Matching (Cosine Similarity)
* 📝 JD Generator (Rule-Based + Claude AI)
* 📊 EDA Dashboard
* 🌐 Streamlit Web App

---

## 🧠 Pipeline

```
Raw Resume
   ↓
Cleaning
   ↓
Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Classification (Centroids)
   ↓
Matching (Cosine Similarity)
   ↓
JD Generation
```

---

## 🔬 NLP Techniques

* Tokenization (NLTK)
* Stopword Removal
* Lemmatization
* TF-IDF
* Cosine Similarity
* N-grams

---

## 🧩 Model Approach

### Centroid-Based Classifier

1. Convert resumes → TF-IDF
2. Compute centroid per category
3. Compare using cosine similarity
4. Pick highest score

✔ Interpretable
✔ Fast
✔ No black-box

---

## 📁 Project Structure

```
resume_nlp_project/
├── app.py
├── cleaner.py
├── preprocessor.py
├── vectorizer.py
├── classifier.py
├── matcher.py
├── jd_generator.py
├── ResumeDataSet.csv
└── README.md
```

---

## ⚙️ Installation

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn nltk
streamlit run app.py
```

---

## 📊 Dataset

* 962 resumes
* 25 categories

| Column   | Description |
| -------- | ----------- |
| Category | Job role    |
| Resume   | Text        |

---

## 🤖 AI Integration

* Claude API for smart JD generation
* Dynamic & realistic outputs
* Optional inside app

---

## 📌 Future Improvements

* Add Accuracy & F1-score
* Add BERT / ML models
* Skill extraction (NER)
* Resume ranking system

---

## 👥 Team

* **Hagar Ayman**
  📧 [hagar.abosamra55@gmail.com](mailto:hagar.abosamra55@gmail.com)
  🔗 https://www.linkedin.com/in/hagar-ayman-ab2a3229a/

* **Hend Elkholy**
  📧 [Hendelkholy55@gmail.com](mailto:Hendelkholy55@gmail.com)
  🔗 https://www.linkedin.com/in/hend-elkholy-32a4ba294

* **Ahmed Khaled Mohamed Mansor**
  📧 [ahmed.ghaith979@gmail.com](mailto:ahmed.ghaith979@gmail.com)
  🔗 https://www.linkedin.com/in/ahmedkhaled-ai
  💻 https://github.com/Ahmedkhaled1122

* **Moustafa Fathalla**
  📧 [moustafa05omar@gmail.com](mailto:moustafa05omar@gmail.com)
  🔗 https://www.linkedin.com/in/moustafafathalla/
  💻 https://github.com/MoustafaFathalla2005

* **Mina Ibrahim**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

## 📝 License

Educational / Portfolio Use

---

## 🔥 Setup Notes

* Add your banner image as: `banner.png`
* Add demo recording as: `demo.gif`
* Place both files in the root directory

---
