# 🎬 VibeCheck — Sentiment Analyser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-89.34%25-brightgreen?style=for-the-badge)

**A Gen Z-friendly ML-powered sentiment analysis web app.**  
Type any review → get instant Positive or Negative prediction with confidence score.

[🚀 Live Demo](#how-to-run) • [📊 Results](#results) • [🛠 Tech Stack](#tech-stack) • [💭 Reflection](#reflection)

</div>

---

## 🧠 Problem

Online platforms receive thousands of reviews daily. Manually reading each one to gauge customer sentiment is **impossible at scale**. There is a clear need for an automated system that instantly classifies text as positive or negative — saving time, enabling data-driven decisions.

---

## ✅ Solution

An end-to-end NLP pipeline trained on **50,000 IMDB movie reviews**:

1. **Preprocessing** — lowercase, HTML tag removal, punctuation stripping, stopword removal
2. **Feature Extraction** — TF-IDF vectorisation with bigrams (`ngram_range=(1,2)`)
3. **Classification** — Logistic Regression binary classifier
4. **Dashboard** — VibeCheck Streamlit UI with live prediction, confidence bars, and history tracker

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.34%** |
| Precision (Negative) | 0.90 |
| Precision (Positive) | 0.88 |
| Recall (Negative) | 0.88 |
| Recall (Positive) | 0.91 |
| F1-Score (Macro Avg) | 0.89 |

> Tested on 10,000 held-out reviews (20% split)

---

## 🖥️ VibeCheck Dashboard

- 🎨 **Purple gradient UI** — modern Gen Z aesthetic
- ⚡ **Instant predictions** — type any review, get result in milliseconds
- 📊 **Confidence bars** — see positive vs negative probability
- 🕒 **History tracker** — last 5 analyses shown
- 💡 **Quick chips** — one-click sample reviews to try

---

## 🗂️ Project Structure

```
sentiment-analyser/
├── data/
│   └── IMDB Dataset.csv        # 50,000 labelled movie reviews
├── notebooks/
│   └── sentiment_analysis.ipynb  # Full EDA + model training notebook
├── outputs/
│   ├── class_distribution.png   # Sentiment class balance plot
│   └── confusion_matrix.png     # Model evaluation heatmap
├── src/
│   └── preprocess.py            # Text preprocessing module
├── app.py                       # VibeCheck Streamlit dashboard
├── model.pkl                    # Trained Logistic Regression model
├── tfidf.pkl                    # Fitted TF-IDF vectoriser
├── requirements.txt             # Dependencies
└── reflection.md                # Learning reflection
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| pandas | Data loading & manipulation |
| NLTK | Stopword removal |
| scikit-learn | TF-IDF + Logistic Regression + evaluation |
| matplotlib / seaborn | Visualisations |
| Streamlit | Web dashboard |
| pickle | Model serialisation |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Nyctophileeeee/sentiment-analyser.git
cd sentiment-analyser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Download [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB Dataset.csv` in the `data/` folder.

### 4. Run the dashboard
```bash
streamlit run app.py
```

### 5. Or run the notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

> **Note:** On first launch, the app auto-trains the model (~30 seconds) and saves `model.pkl` and `tfidf.pkl` for instant loading next time.

---

## 💭 Reflection

This project taught me how to build a complete NLP pipeline from raw messy text to a deployed, interactive ML application. Key learnings:

- How **TF-IDF** converts text into numerical features machines understand
- The difference between `fit_transform` (training) and `transform` (inference)
- How **Logistic Regression** works for binary classification with probability outputs
- How to deploy an ML model as a real web app using **Streamlit**
- Real-world challenges: kernel setup, library conflicts, CSS in Streamlit

The 89.34% accuracy on 50,000 reviews demonstrates that classical ML methods remain powerful for NLP tasks — you don't always need a large language model.

---

## 📁 Dataset

- **Name:** IMDB Dataset of 50K Movie Reviews
- **Source:** [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews (25,000 positive, 25,000 negative)
- **Labels:** `positive` / `negative`

---

## 👤 Author

**Ayush** — BSc Applied Artificial Intelligence, University of Bradford  
📎 [GitHub](https://github.com/Nyctophileeeee) • [LinkedIn](https://linkedin.com/in/your-profile)

---

<div align="center">
Made with 💜 as part of COS6031-D Applied AI Professional Portfolio
</div>
