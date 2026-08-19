# Demo Guide

1. Open `StudyMate.ipynb` in Google Colab (or locally with `jupyter notebook`).
2. Run all cells. The dataset cell streams only the first 20,000 records, so it is fast.
3. Show the dataset size and sample.
4. Run TF-IDF baseline evaluation on the held-out test split.
5. Run Sentence Transformer evaluation and show the confusion matrix.
6. Show Accuracy, Precision, Recall and F1 (test split), plus the accuracy improvement.
7. Run the StudyMate weak-topic demo: it extracts clean topics, evaluates the learner's
   answers semantically, and prints personalized revision recommendations.
8. Open the Streamlit frontend for the interactive presentation:
   `streamlit run frontend/app.py`.

## What to tell the faculty

"MS MARCO gives us a large-scale QA resource. We compare a traditional TF-IDF baseline
with a transformer-based semantic model, evaluated on a held-out test split with
thresholds chosen on training data. The final StudyMate layer maps answer performance
back to study topics so the learner gets targeted recommendations."
