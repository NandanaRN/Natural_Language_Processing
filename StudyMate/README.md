# StudyMate — AI Learning Assistant

> **An NLP-based learning assistant for semantic answer evaluation, weak-topic detection, and personalized revision recommendations.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-6C5CE7?style=for-the-badge&logo=streamlit&logoColor=white)](https://naturallanguageprocessing-jjhfcvdl5acyfw8vju4e7i.streamlit.app)

StudyMate is an **NLP mini-project** designed to help students identify their learning gaps and receive focused revision guidance.

Instead of functioning only as a question-answering chatbot, StudyMate focuses on **learner profiling at the topic level**. It processes study material, evaluates student answers using semantic similarity, detects weak areas, and recommends topics for revision.

---

## Key Features

* Study material processing
* Topic extraction
* Student answer evaluation
* Semantic similarity analysis
* Weak topic detection
* Personalized revision recommendations
* TF-IDF baseline comparison
* Sentence Transformer-based semantic retrieval
* Accuracy, Precision, Recall, and F1-score evaluation

---

## Project Workflow

```text
Study Material
      ↓
NLP Processing
      ↓
Topic Extraction
      ↓
Answer Evaluation
      ↓
Semantic Similarity
      ↓
Weak Topic Detection
      ↓
Personalized Recommendation
```

**Demo Flow:**
Study Material → NLP Processing → Topic Extraction → Answer Evaluation → Weak Topic Detection → Personalized Recommendation

---

## Dataset

StudyMate uses the **Microsoft MS MARCO Question Answering Dataset v2.1**, a large-scale question-answering dataset containing human-generated answers.

### Dataset Statistics

| Split      |       Queries |
| ---------- | ------------: |
| Training   |       808,731 |
| Validation |       101,093 |
| Test       |       101,092 |
| **Total**  | **1,010,916** |

The complete dataset is approximately **2 GB**, so it is not included in this repository.

The notebook uses **Hugging Face streaming mode** to load the first **20,000 training records** for the lightweight demonstration.

**Dataset:**
https://huggingface.co/datasets/microsoft/ms_marco

---

## NLP Models

### TF-IDF — Baseline

TF-IDF (Term Frequency–Inverse Document Frequency) is used as the traditional lexical similarity baseline.

```text
Student Answer
      ↓
TF-IDF
      ↓
Vector Representation
      ↓
Cosine Similarity
      ↓
Similarity Score
```

### Sentence Transformer — Improved Model

The project uses **`all-MiniLM-L6-v2`** to generate sentence embeddings and measure semantic similarity.

```text
Student Answer
      ↓
Sentence Transformer
      ↓
Semantic Embedding
      ↓
Cosine Similarity
      ↓
Similarity Score
```

This helps the system identify answers with similar meanings even when different words are used.

---

## Methodology

1. Load MS MARCO QA v2.1 using Hugging Face streaming.
2. Create a lightweight sample of 20,000 records.
3. Prepare question and answer text.
4. Apply TF-IDF as the baseline approach.
5. Generate semantic embeddings using `all-MiniLM-L6-v2`.
6. Calculate cosine similarity scores.
7. Generate weak/strong answer proxy labels.
8. Select the classification threshold using the training split based on the best F1-score.
9. Evaluate both approaches on a held-out test split.
10. Identify weak topics and generate revision recommendations.

---

## Evaluation

The notebook evaluates both approaches using:

* Accuracy
* Precision
* Recall
* F1-score
* Accuracy improvement over baseline

---

## Why StudyMate?

Traditional keyword-based approaches may struggle when two answers have similar meanings but use different vocabulary.

For example:

```text
Reference:
"Machine learning enables computers to learn patterns from data."

Student:
"ML allows systems to identify patterns and improve using available data."
```

A semantic model can recognize the similarity in meaning even though the wording is different.

StudyMate uses this semantic understanding for:

**Answer Evaluation → Learner Profiling → Weak Topic Detection → Personalized Revision**

---

## Project Contribution

The main contribution of StudyMate is combining **semantic answer evaluation with topic-level learner profiling**.

Instead of only providing an answer, StudyMate aims to identify:

> **Which topics does the student need to revise?**

This positions the project as a **personalized learning assistant** rather than only a conventional question-answering or PDF chatbot.

---

## Technology Stack

| Technology            | Purpose                               |
| --------------------- | ------------------------------------- |
| Python                | Core development                      |
| NLP                   | Text processing and analysis          |
| Scikit-learn          | TF-IDF, cosine similarity, evaluation |
| Sentence Transformers | Semantic embeddings                   |
| Hugging Face Datasets | Dataset loading and streaming         |
| Pandas                | Data processing                       |
| NumPy                 | Numerical computation                 |
| Streamlit             | Interactive frontend                  |
| Jupyter Notebook      | Experimentation and evaluation        |

---

## Project Structure

```text
StudyMate/
│
├── StudyMate.ipynb
├── README.md
├── requirements.txt
│
└── frontend/
    └── app.py
```

---

## Installation

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd StudyMate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Notebook

```bash
jupyter notebook StudyMate.ipynb
```

The notebook performs:

```text
Dataset Loading
      ↓
20K Sample Creation
      ↓
Train/Test Split
      ↓
TF-IDF Baseline
      ↓
Sentence Transformer
      ↓
Threshold Selection
      ↓
Held-out Evaluation
      ↓
Metric Comparison
```

The notebook can also be opened and executed using **Google Colab**.

---

## Run the Streamlit Application

From the project root:

```bash
streamlit run frontend/app.py
```

---

## Research Gap

Many learning assistants primarily focus on answering questions or retrieving information.

StudyMate focuses on **identifying learning weaknesses and recommending what the student should revise next**.

The project combines:

**Semantic NLP + Answer Evaluation + Weak Topic Detection + Personalized Recommendation**

---

## Limitations

* The demonstration uses a 20,000-record sample instead of the complete MS MARCO dataset.
* Proxy labels may not perfectly represent real student performance.
* MS MARCO is a general QA dataset and is not specifically designed for educational assessment.
* Topic extraction can be improved using domain-specific educational datasets.
* A manually labelled student-answer dataset would provide stronger evaluation.
* The current system focuses on topic-level revision rather than long-term adaptive learning.

---

## Future Scope

* Domain-specific educational datasets
* Expert-labelled student-answer datasets
* BERT-based answer assessment
* Student performance dashboards
* Continuous learner profiling
* Advanced personalized recommendation
* PDF and lecture-note ingestion
* Explainable feedback
* Learning progress tracking

---

## Reference

**Redhu, V., Singh, A. K., & Saravanan, M.**
*AI-Enhanced Learning Assistant Platform.* 2024.

---

## Author

**Nandana R. Nair**
MCA — Generative AI

---

## License

This project is developed for **academic and educational purposes**.
