# Text Paraphrasing via Matrix Decomposition

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-1.0+-green.svg)](https://flask.palletsprojects.com/)

A text re-structuring and paraphrasing system using matrix decomposition (NMF/LDA) for topic modeling and sentence clustering.

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> Overview

This project implements an unsupervised approach to text paraphrasing and restructuring. It uses matrix decomposition techniques to:

1. **Enumerate topics** from raw text without supervision
2. **Evaluate topic quality** using custom metrics (average map rate and continuity rate)
3. **Re-structure text** into context-sensitive groups

## <img src=".github/icons/rocket.svg" width="16" height="16" alt="rocket"> Quick Start

### Installation

```bash
pip install flask scikit-learn numpy pandas jieba nltk
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run the Web Server

```bash
python re_paraphrasing.py
```

Then open your browser to `http://localhost:5000`

## <img src=".github/icons/folder.svg" width="16" height="16" alt="folder"> Project Structure

```
text_Paraphrasing/
├── re_paraphrasing.py           # Main application with Flask server
├── templates/
│   └── home.html                # Web interface
├── vector.pkl                   # Cached vectorizer (auto-generated)
├── re_structured_text.txt       # Sample output
├── topic_evaluate.txt           # Evaluation results
├── demo.png                     # Demo screenshot
└── README.md                    # This file
```

## 🧠 Algorithm Overview

### 1. Topic Enumeration

Uses unsupervised matrix decomposition to discover topics:
- **NMF** (Non-negative Matrix Factorization): Good for interpretable topics
- **LDA** (Latent Dirichlet Allocation): Probabilistic topic model

### 2. Quality Metrics

Two custom metrics evaluate topic quality:

| Metric | Description |
|--------|-------------|
| **Average Map Rate** | How close topic frequent words are to each sentence (semantic relevance) |
| **Continuity Rate** | How well each topic forms context-sensitive sentences (coherence) |

### 3. Grid Search for Optimal Topics

The algorithm searches for the optimal number of topics by maximizing:
```
Score = Average Map Rate + Continuity Rate
```

### 4. Re-paraphrasing

Groups sentences by topic and orders them by continuity to create readable, context-sensible groups.

## <img src=".github/icons/chart.svg" width="16" height="16" alt="chart"> Example Output

```
Topic #4: 愿意 准备 焦虑 到底 孩子 知道 手机 不行 我要 辞职
Topic #5: 这种 知道 情况 孩子 需要 不想 踏踏实实 之后 照顾 请问

Re-paraphrased with highest continuity:
481 我，我觉得我一直找不到特别喜欢的人。
482 找不到特别喜欢的
484 是你从来没有遇到过自己，特别喜欢的还是说遇到过，喜欢的...
```

## 🌐 API Usage

### Web Interface

Send a POST request to `/answer`:

```bash
curl -X POST http://localhost:5000/answer \
  -H "Content-Type: application/json" \
  -d '{"passage": "your text here", "question": "optional question"}'
```

Response:
```json
{
  "answer": "re-structured text with topic summaries..."
}
```

### Python API

```python
from re_paraphrasing import paraphrase

result = paraphrase(
    texts="your text here",
    n_components=25,
    mode='demo',
    sent_tokenize='。'
)
print(result)
```

## 📝 Algorithm Details

### Preprocessing

1. Chinese text segmentation using jieba
2. Remove numbers and English characters
3. TF-IDF vectorization

### Matrix Factorization

```python
# NMF with KL divergence
nmf = NMF(
    n_components=n_components,
    beta_loss='kullback-leibler',
    solver='mu',
    max_iter=1000
)

# Factorization: V ≈ W × H
# W: Document-topic matrix
# H: Topic-word matrix
```

### Inference

For each document:
1. Find the dominant topic (argmax of W row)
2. Calculate map rate: overlap between document words and topic words
3. Group documents by topic
4. Calculate continuity: average adjacency of documents in same topic

## <img src=".github/icons/warning.svg" width="16" height="16" alt="warning"> Configuration

Update the stopwords path in `re_paraphrasing.py`:

```python
STOPWORDS_PATH = 'path/to/your/stopwords.txt'
```

## <img src=".github/icons/wrench.svg" width="16" height="16" alt="wrench"> Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_df` | 0.95 | Ignore terms with document frequency > max_df |
| `min_df` | 2 | Ignore terms with document frequency < min_df |
| `max_features` | 3000 | Maximum vocabulary size |
| `n_components` | 30 | Number of topics (auto-adjusted for small texts) |
| `n_top_words` | 10 | Number of words per topic |

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> References

- [Non-negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
- [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## <img src=".github/icons/image.svg" width="16" height="16" alt="image"> Demo

![Server Demo](demo.png)
