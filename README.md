# Low-Resource Customer Feedback Sentiment Analysis

---

# Project Overview

This project presents a low-resource sentiment analysis system developed using multiple open-source transformer-based language models. The objective is to build an accurate customer sentiment classifier using only **1,000 labelled training samples**, while proposing an efficient strategy for utilizing a much larger collection of unlabelled customer feedback through semi-supervised learning.

Several state-of-the-art transformer architectures were evaluated, including DistilBERT, BERT, RoBERTa, MiniLM, ELECTRA, and multiple DeBERTa variants. The project follows a modular implementation consisting of dataset preprocessing, model training, evaluation, experiment tracking, and reproducible model management.

---

# Objectives

- Build a sentiment classifier using only the labelled customer feedback.
- Compare the performance of multiple transformer models.
- Evaluate each model using standard classification metrics.
- Design a strategy to leverage unlabelled data through pseudo-labeling.
- Develop a reproducible and scalable experimentation pipeline.
- Balance model performance with computational efficiency.

---

# Project Structure

```text
Low-Resource-Customer-Feedback-Sentiment-Analysis/

├── data/
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── unlabelled.csv
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── outputs/
│   ├── bert-base-uncased/
│   ├── distilbert-cpu/
│   ├── google_electra-base-discriminator/
│   ├── microsoft_MiniLM-L12-H384-uncased/
│   ├── microsoft_deberta-v3-small/
│   ├── microsoft_deberta-v3-base/
│   ├── microsoft_deberta-v3-large/
│   └── roberta-base/
│
├── figures/
├── sentiment_classifier.ipynb
├── requirements.txt
└── README.md
```

---

# Dataset

The dataset consists of four files:

| File | Description |
|------|-------------|
| train.csv | Labelled training dataset |
| valid.csv | Validation dataset |
| test.csv | Test dataset |
| unlabelled.csv | Unlabelled customer feedback |

Each labelled sample contains:

- text
- sentiment

The unlabelled dataset contains only customer feedback text and is intended for pseudo-labeling and semi-supervised learning.

---

# Models Evaluated

The following transformer architectures were evaluated:

- DistilBERT
- BERT Base
- RoBERTa Base
- MiniLM-L12-H384
- ELECTRA Base Discriminator
- DeBERTa-v3 Small
- DeBERTa-v3 Base
- DeBERTa-v3 Large

---

# Experimental Pipeline

1. Load the labelled and unlabelled datasets.
2. Tokenize customer feedback using Hugging Face tokenizers.
3. Fine-tune transformer models on the labelled training set.
4. Evaluate models on the validation dataset.
5. Select and save the best-performing checkpoint.
6. Evaluate on the independent test dataset.
7. Compare multiple transformer architectures.
8. Propose pseudo-labeling for future semi-supervised learning.

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/divziv/Low-Resource-Customer-Feedback-Sentiment-Analysis.git

cd Low-Resource-Customer-Feedback-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Train a model:

```bash
python src/train.py
```

Evaluate the trained model:

```bash
python src/evaluate.py
```

Run the notebook:

```bash
jupyter notebook sentiment_classifier.ipynb
```

---

# Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Validation Results

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|-------:|---------:|
| DistilBERT | **86.40%** | 90.38% | 93.07% | 91.71% |
| BERT Base | **90.00%** | 97.40% | 90.36% | 93.75% |
| RoBERTa Base | **91.00%** | 96.10% | 92.50% | 94.27% |
| MiniLM-L12-H384 | **93.00%** | 96.10% | 94.87% | 95.48% |
| ELECTRA Base | **93.00%** | 96.10% | 94.87% | 95.48% |
| DeBERTa-v3 Small | **93.00%** | 94.81% | 96.05% | 95.42% |
| DeBERTa-v3 Base | **91.00%** | 93.51% | 94.74% | 94.12% |
| **DeBERTa-v3 Large** | **94.00%** | **96.10%** | **96.10%** | **96.10%** |

---

# Test Results

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|-------:|---------:|
| MiniLM-L12-H384 | **85.20%** | 90.59% | 91.04% | 90.82% |
| DistilBERT | **86.40%** | 90.38% | 93.07% | 91.71% |
| DeBERTa-v3 Base | **87.60%** | 92.57% | 92.12% | 92.35% |
| ELECTRA Base | **87.60%** | 94.55% | 90.52% | 92.49% |
| BERT Base | **88.40%** | 97.03% | 89.50% | 93.11% |
| RoBERTa Base | **88.80%** | 96.04% | 90.65% | 93.27% |
| DeBERTa-v3 Small | **90.80%** | 95.54% | 93.24% | 94.38% |
| **DeBERTa-v3 Large** | **92.40%** | **97.03%** | **93.78%** | **95.38%** |

---

# Overall Model Ranking

| Rank | Model | Test Accuracy | Test F1 Score |
|------|-------------------------------|-------------:|--------------:|
| 🥇 1 | **DeBERTa-v3 Large** | **92.40%** | **95.38%** |
| 🥈 2 | **DeBERTa-v3 Small** | **90.80%** | **94.38%** |
| 🥉 3 | **RoBERTa Base** | **88.80%** | **93.27%** |
| 4 | **BERT Base** | **88.40%** | **93.11%** |
| 5 | **ELECTRA Base** | **87.60%** | **92.49%** |
| 6 | **DeBERTa-v3 Base** | **87.60%** | **92.35%** |
| 7 | **DistilBERT** | **86.40%** | **91.71%** |
| 8 | **MiniLM-L12-H384** | **85.20%** | **90.82%** |

---

# Key Findings

- DeBERTa-v3 Large achieved the highest performance across all evaluated models.
- DeBERTa-v3 Small provided the best balance between computational efficiency and predictive performance.
- DistilBERT trained considerably faster and required fewer computational resources, making it suitable for rapid prototyping.
- MiniLM achieved strong validation performance but exhibited a larger drop on the independent test set, indicating comparatively weaker generalization.
- Larger transformer architectures generally produced better sentiment classification performance but required substantially greater computational resources and training time.

---

# Challenges Encountered

## Computational Challenges

- CPU-only local development
- Long training time for large transformer models
- GPU memory limitations
- Mixed precision (FP16) instability

## Model Challenges

- Hyperparameter sensitivity
- Classification head initialization
- Training instability
- Hugging Face checkpoint compatibility warnings

## Dataset Challenges

- Limited labelled dataset
- Low-resource learning scenario
- Class imbalance
- Generalization to unseen customer feedback

## Experiment Management

- Managing outputs from multiple models
- Reproducibility
- Large checkpoint storage
- GitHub file size limitations

---

# Proposed Semi-Supervised Learning Strategy

To leverage the unlabelled customer feedback dataset:

1. Train an initial classifier using the labelled data.
2. Predict labels for the unlabelled samples.
3. Select only high-confidence predictions.
4. Merge pseudo-labelled samples with the original labelled dataset.
5. Retrain the model using the expanded dataset.
6. Repeat until validation performance converges.

This approach reduces manual annotation effort while improving the model's ability to learn from a much larger dataset.

---

# Future Improvements

- Semi-supervised learning
- Active learning
- Confidence-based pseudo-labeling
- LoRA / PEFT fine-tuning
- Hyperparameter optimization
- Knowledge distillation
- Model ensembling
- Cross-validation
- Automatic threshold optimization

---

# Repository Contents

Each experiment directory contains:

- Trained model checkpoint
- Tokenizer
- Configuration files
- Evaluation metrics
- Predictions
- Confusion matrix
- Training loss curve
- Validation accuracy curve

---

# Conclusion

This project investigated multiple open-source transformer architectures for sentiment analysis under a low-resource setting using only **1,000 labelled customer feedback samples**. Despite the limited amount of labelled data, all transformer-based approaches achieved strong performance, demonstrating the effectiveness of transfer learning for text classification.

Among the evaluated models, **DeBERTa-v3 Large** consistently achieved the best overall performance, obtaining **94.00% validation accuracy**, **92.40% test accuracy**, and a **95.38% F1-score**. Its stronger contextual representation enabled better generalization than the other evaluated architectures, making it the strongest model for this task.

From a practical deployment perspective, **DeBERTa-v3 Small** offered the best balance between predictive performance and computational efficiency, achieving **90.80% test accuracy** and a **94.38% F1-score** while requiring substantially fewer computational resources than the Large variant. This makes it a practical choice for resource-constrained environments.

The experiments also highlighted the trade-off between model complexity and computational cost. Smaller models such as **DistilBERT** trained significantly faster and were easier to debug, whereas larger transformer architectures required careful hyperparameter tuning, GPU acceleration, and more sophisticated experiment management. Furthermore, the reduced test performance observed for **MiniLM** emphasizes the importance of evaluating models on an independent test dataset rather than relying solely on validation performance.

Overall, this work demonstrates that **open-source transformer models can achieve excellent sentiment classification performance under limited labelled data conditions**. Future work will focus on incorporating the unlabelled dataset through pseudo-labeling and semi-supervised learning to further improve classification accuracy while minimizing manual annotation effort.

---

# License

This repository was developed purely intended for educational and research purposes.
