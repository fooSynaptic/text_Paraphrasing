"""
Text Paraphrasing via Matrix Decomposition

This module implements text re-structuring and paraphrasing using:
- NMF (Non-negative Matrix Factorization) or LDA (Latent Dirichlet Allocation)
- Custom metrics: average map rate and continuity rate
- Flask web server for interactive use

Example:
    >>> from re_paraphrasing import paraphrase
    >>> result = paraphrase("Your text here", n_components=25)
"""

import os
import re
import json
import copy
import random
from time import time
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from jieba import cut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Configuration
STOPWORDS_PATH = 'path/to/your/own/stopwords.txt'
VECTOR_CACHE_PATH = 'vector.pkl'

# Initialize Flask app
template_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(template_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# Initialize stopwords
try:
    with open(STOPWORDS_PATH, encoding='utf-8') as f:
        STOPWORDS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    STOPWORDS = []

# Initialize or load vectorizer
if os.path.exists(VECTOR_CACHE_PATH):
    import joblib
    VECTORIZER = joblib.load(VECTOR_CACHE_PATH)
else:
    VECTORIZER = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=3000,
        stop_words=STOPWORDS
    )


def print_top_words(model, feature_names: List[str], n_top_words: int) -> Dict[int, List[str]]:
    """
    Extract top words for each topic.
    
    Args:
        model: Fitted NMF or LDA model
        feature_names: List of feature names
        n_top_words: Number of top words to extract
        
    Returns:
        Dictionary mapping topic index to list of top words
    """
    topic_dict = {}
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        topic_dict[topic_idx] = [feature_names[i] for i in top_indices]
    
    return topic_dict


def factorize_matrix(
    matrix: np.ndarray,
    n_components: int,
    feature_names: List[str],
    n_top_words: int,
    factor_method: str = "LDA"
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """
    Factorize the document-term matrix using NMF or LDA.
    
    Args:
        matrix: Document-term matrix (TF-IDF or count)
        n_components: Number of topics
        feature_names: List of feature names
        n_top_words: Number of words per topic
        factor_method: "NMF" or "LDA"
        
    Returns:
        Tuple of (W matrix, topic dictionary)
        
    Raises:
        ValueError: If factor_method is not "NMF" or "LDA"
    """
    t0 = time()
    
    if factor_method == "NMF":
        model = NMF(
            n_components=n_components,
            random_state=1,
            beta_loss='kullback-leibler',
            solver='mu',
            max_iter=1000,
            alpha=.1,
            l1_ratio=.5
        ).fit(matrix)
    elif factor_method == 'LDA':
        model = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=5,
            learning_method='online',
            learning_offset=50.,
            random_state=0
        ).fit(matrix)
    else:
        raise ValueError(f"Unknown factor method: {factor_method}")
    
    print(f"Factorization completed in {time() - t0:.3f}s")
    
    topic_dict = print_top_words(model, feature_names, n_top_words)
    W = model.transform(matrix)
    
    return W, topic_dict


def calculate_metrics(
    W: np.ndarray,
    topics: Dict[int, List[str]],
    vocab: set,
    corpus: List[str]
) -> Tuple[float, float, Dict[int, List[int]], List[str]]:
    """
    Calculate map rate and continuity rate metrics.
    
    Args:
        W: Document-topic matrix
        topics: Dictionary of topic words
        vocab: Vocabulary set
        corpus: Original corpus
        
    Returns:
        Tuple of (avg_map_rate, continuity_rate, topic_assignments, summary)
    """
    map_rates = []
    topic_assignments = defaultdict(list)
    
    # Calculate map rates and assign documents to topics
    for doc_idx, doc_distribution in enumerate(W):
        dominant_topic = np.argmax(doc_distribution)
        topic_words = topics[dominant_topic]
        
        # Count how many topic words appear in document
        doc_tokens = set()
        for word in vocab:
            if word in corpus[doc_idx]:
                doc_tokens.add(word)
        
        matching_tokens = [w for w in doc_tokens if w in topic_words]
        map_rate = len(matching_tokens) / len(topic_words) if topic_words else 0
        map_rates.append(map_rate)
        
        topic_assignments[dominant_topic].append(doc_idx)
    
    # Calculate continuity rate
    continuity_scores = {}
    for topic_id, doc_indices in topic_assignments.items():
        if len(doc_indices) < 2:
            continuity_scores[topic_id] = 0
            continue
        
        # Check if consecutive documents are in the same topic
        consecutive_pairs = [
            doc_indices[i] - doc_indices[i - 1] < 3
            for i in range(1, len(doc_indices))
        ]
        continuity_scores[topic_id] = sum(consecutive_pairs) / len(consecutive_pairs)
    
    # Calculate average continuity (excluding topic 0 which is often noise)
    valid_topics = [t for t in continuity_scores.keys() if t != 0]
    avg_continuity = sum(continuity_scores[t] for t in valid_topics) / len(valid_topics) if valid_topics else 0
    
    avg_map_rate = sum(map_rates) / len(map_rates) if map_rates else 0
    
    # Generate re-structured text
    restructured = []
    top_topics = [t[0] for t in Counter(topic_assignments).most_common(5) if t[0] != 0]
    
    for topic_id in top_topics:
        section = "\n\n"
        section += f"Topic {topic_id}\n\n\n"
        
        for doc_idx in topic_assignments[topic_id]:
            cleaned_text = corpus[doc_idx].replace(' ', '')
            section += f"{doc_idx}\t{cleaned_text}\n"
        
        restructured.append(section)
    
    summary = (
        f"The average performance of this text re-structure result "
        f"TOPIC MAP RATE as: {avg_map_rate:.4f} and "
        f"CONTINUITY RATE as: {avg_continuity:.4f}..."
    )
    restructured.insert(0, summary)
    
    return avg_map_rate, avg_continuity, dict(topic_assignments), restructured


def paraphrase(
    texts: str,
    n_components: int,
    mode: str = 'demo',
    sent_tokenize: str = '。'
) -> str:
    """
    Paraphrase and re-structure text using matrix decomposition.
    
    Args:
        texts: Input text or path to text file
        n_components: Number of topics (adjusted for small texts)
        mode: 'demo' for prepared text, otherwise direct input
        sent_tokenize: Sentence delimiter
        
    Returns:
        Re-structured text with topic summaries
        
    Example:
        >>> result = paraphrase("Your text here", n_components=25)
        >>> print(result)
    """
    global corpus
    
    # Load corpus
    if mode == 'demo':
        path = 'your_prepared_text.txt'
        try:
            corpus = [item['text'] for item in json.loads(open(path).read())]
        except FileNotFoundError:
            corpus = [texts]
    else:
        corpus = [sent.strip() for sent in texts.split(sent_tokenize) if sent.strip()]
    
    # Preprocess
    sents = [re.sub(r'[0-9 a-z]', '', text) for text in corpus]
    sents = [' '.join(cut(x)) for x in sents]
    
    # Build vocabulary
    vocab = set()
    for text in sents:
        vocab.update(text.split())
    
    # Vectorize
    if os.path.exists(VECTOR_CACHE_PATH):
        feature_matrix = VECTORIZER.transform(sents).toarray()
    else:
        feature_matrix = VECTORIZER.fit_transform(sents).toarray()
        import joblib
        joblib.dump(VECTORIZER, VECTOR_CACHE_PATH)
    
    # Determine number of components
    if len(sents) <= 30:
        n_components = min(2, len(sents) // 2)
    else:
        n_components = min(n_components, len(sents) // 3)
    
    # Factorize
    W, topics = factorize_matrix(
        feature_matrix,
        n_components,
        VECTORIZER.get_feature_names_out(),
        10,
        factor_method="NMF"
    )
    
    # Calculate metrics and generate output
    _, _, _, result = calculate_metrics(W, topics, vocab, corpus)
    
    return '\n'.join(result)


# Flask routes
@app.route("/")
def home():
    """Render home page."""
    return render_template('home.html')


@app.route('/answer', methods=['POST'])
def answer():
    """
    API endpoint for text paraphrasing.
    
    Expects JSON with 'passage' field.
    Returns JSON with 'answer' field containing re-structured text.
    """
    passage = request.json.get('passage', '')
    question = request.json.get('question', '')
    
    result = paraphrase(passage, n_components=25, mode='input', sent_tokenize='。')
    
    print(f"Received response: {result[:100]}...")
    response = {"answer": result}
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=True)
