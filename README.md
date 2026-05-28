# Transformers for Suicidal Ideation Detection in Reddit Posts

> **UC Berkeley MIDS W266: Natural Language Processing with Deep Learning** · Spring 2025
> Solo research project. Full paper, presentation, and notebooks included.

Evaluating BERT, RoBERTa, and BERTweet for detecting suicidal ideation in Reddit posts — benchmarked against a TF-IDF + Logistic Regression baseline, with deep analysis of misclassifications, interpretability, and clinical deployment considerations.

📄 [Read the paper](./paper/Transformers_for_Suicidal_Ideation_Detection_in_Reddit_Posts.pdf) · 🎯 [Presentation deck](./presentation/)

---

## Motivation

Suicide is among the top ten causes of death globally and the second leading cause of death among 15–29-year-olds. Anonymous platforms like Reddit (r/SuicideWatch, r/depression, r/teenagers) have become spaces where people disclose distress in ways traditional clinical screening can't capture — often through metaphor, sarcasm, or veiled language that rule-based and classical ML approaches miss.

This project asks: **can transformer architectures reliably detect indirect, nuanced expressions of suicidal ideation in a way that could augment, not replace, clinical screening?** And critically, can they do so while remaining interpretable, fair, and clinically actionable?

---

## Approach

**Dataset.** 232,000+ labeled Reddit posts (Kaggle Suicide Watch dataset), balanced 50/50 across suicide vs. non-suicide classes for fair comparison.

**Models trained and compared (all implemented from scratch):**
- `TF-IDF + Logistic Regression` — classical baseline
- `BERT (bert-base-uncased)` — vanilla transformer baseline
- `RoBERTa (roberta-base)` — optimized pretraining
- `BERTweet (vinai/bertweet-base)` — pretrained on social media text

**Methodology highlights:**
- Strict class balance maintained across all train/val splits
- HuggingFace Transformers + PyTorch Trainer for fine-tuning
- SHAP analysis and attention visualization for interpretability
- Dedicated misclassification analysis surfacing failure modes on metaphorical and sarcastic content
- Drew on 13 peer-reviewed sources to ground methodology in prior SID literature

---

## Results

All transformer models outperformed the TF-IDF baseline (93% F1). RoBERTa and BERTweet both reached **~98% macro F1** on the balanced 10K evaluation subset:

| Model | Suicide-class Recall | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | 0.92 | Baseline |
| BERT | ~0.96 | Lower precision; pretraining corpus (BookCorpus + Wikipedia) doesn't match Reddit register |
| **RoBERTa** | **0.992** | Best recall — critical for minimizing false negatives in clinical use |
| BERTweet | ~0.98 | Best precision; strong on emojis, abbreviations, informal phrasing |

**Key insight:** RoBERTa's recall advantage matters disproportionately here — in a clinical screening context, a missed suicidal post is a far more consequential error than a false alarm.

---

## Beyond Accuracy

The paper goes deep on what makes a transformer model *trustworthy* for mental health use, not just accurate:

- **Misclassification analysis** — surfaced patterns in false positives (e.g. dark humor, song lyrics) and false negatives (veiled or metaphorical ideation)
- **Interpretability** — SHAP and attention visualizations showed which tokens drove predictions, and where models leaned on spurious lexical cues
- **Fairness and bias considerations** — discussion of how subreddit-derived labels introduce noise and demographic skew
- **Deployment challenges** — interpretability tradeoffs, latency in real-time monitoring, and the necessity of human-in-the-loop frameworks
- **Future directions** — multilingual adaptation, time-sensitive temporal modeling, and clinician-AI hybrid systems

---

## Repo Structure

\`\`\`
.
├── paper/
│   └── Transformers_for_Suicidal_Ideation_Detection_in_Reddit_Posts.pdf
├── notebooks/
│   ├── Baseline_TFIDF_LogReg.ipynb              # Classical baseline
│   ├── FineTuned_BERT_SuicideDetection.ipynb    # BERT 3-epoch full fine-tune
│   ├── SuicideDetection_BERT_vs_RoBERTa.ipynb   # Head-to-head model comparison
│   ├── BERTweet_SuicideDetection.ipynb          # Social-media pretrained model
│   ├── BERT_Finetune_SHAP.ipynb                 # Interpretability via SHAP
│   ├── BERT_Misclassification.ipynb             # Failure mode analysis
│   └── SuicideDetection_Visualization_Final.ipynb  # Final comparison + visuals
├── presentation/
│   └── W266_Presentation.pdf
└── README.md
\`\`\`

---

## Tech

`Python` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `SHAP` · `pandas` · `matplotlib` · `seaborn` · `Google Colab (GPU)`

---

## A Note on Responsible Use

This work is academic research evaluating model capability and limitations. Real-world deployment of any suicidal ideation detection system requires clinical oversight, ethical review, privacy protections, and explicit human-in-the-loop intervention pathways — none of which a standalone model can provide. The paper discusses these constraints in detail.

If you or someone you know is struggling, the **988 Suicide & Crisis Lifeline** (US) is available 24/7 by call or text.
