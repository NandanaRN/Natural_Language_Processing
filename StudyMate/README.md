# StudyMate — AI Learning Assistant

## NLP Mini Project Demo

StudyMate identifies important topics from study material, evaluates student answers using semantic similarity, detects weak areas, and recommends focused revision.

### Dataset

The project uses **Microsoft MS MARCO QA v2.1**. The dataset has 808,731 training queries, 101,093 validation queries and 101,092 test queries — more than one million queries overall. It is a large-scale question-answering dataset with human-generated answers.

The full dataset is **not included in this ZIP** because the v2.1 download is about 2 GB. The notebook uses Hugging Face **streaming mode** so it downloads only the first 20,000 training records (a small fraction of the first shard), keeping the demo light and fast. For the final experiment, the sample size can be increased.

Dataset: https://huggingface.co/datasets/microsoft/ms_marco

### Models

**Baseline:** TF-IDF + cosine similarity

**Improved:** Sentence Transformer (`all-MiniLM-L6-v2`) + cosine similarity

The notebook calculates, on a **held-out test split**:
- Accuracy
- Precision
- Recall
- F1-score
- Accuracy improvement over baseline

Thresholds are selected on the training split (best F1), so no test-set tuning is done.

### Important accuracy note

Do not claim an accuracy value before running the notebook. The notebook generates the measured values on a held-out test set. The weak/strong labels are automatically generated proxy labels (truncated reference answers); for a final academic paper, use a manually labelled student-answer test set rather than relying only on these proxies.

### Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook StudyMate.ipynb
```

Or open it in Google Colab.

### Run the frontend

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

### Demo flow

Study Material → NLP Processing → Topic Extraction → Answer Evaluation → Weak Topic Detection → Personalized Recommendation

### Reference paper

Vivek Redhu, Abhishek Kumar Singh, M. Saravanan, “AI-Enhanced Learning Assistant Platform,” 2024.

### Project positioning

StudyMate focuses on topic-level learner profiling rather than being only a PDF chatbot. Its main NLP contribution is semantic answer evaluation combined with weak-topic detection and personalized learning recommendations.
