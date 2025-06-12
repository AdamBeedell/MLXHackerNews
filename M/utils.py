import pandas as pd
from sqlalchemy import create_engine
import re
from datetime import datetime
from collections import Counter
import numpy as np
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_hn_data(limit=10000, min_score=10):
    # Connects to the Hacker News database and loads a sample of posts
    # Filters: title must exist, post must be a story, not dead, and meet score threshold
    engine = create_engine('postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki')

    query = f'''
    SELECT id, title, url, by, score, time
    FROM hacker_news.items
    WHERE title IS NOT NULL
      AND score >= {min_score}
      AND dead IS NULL
      AND type = 'story'
    ORDER BY RANDOM()
    LIMIT {limit};
    '''
    return pd.read_sql(query, engine)

def clean_titles(df):
    # Makes all titles lowercase and removes punctuation
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return df

def extract_time_features(df):
    # Converts timestamp into datetime and extracts:
    # - day of the week (0=Monday)
    # - hour of the day (0–23)
    # Then one-hot encodes those into new columns
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['day_of_week'] = df['time'].dt.dayofweek
    df['hour_of_day'] = df['time'].dt.hour

    dow_ohe = pd.get_dummies(df['day_of_week'], prefix='dow')
    hod_ohe = pd.get_dummies(df['hour_of_day'], prefix='hour')

    return pd.concat([df, dow_ohe, hod_ohe], axis=1)

def tokenize_titles(df):
    # Splits titles into lists of words using NLTK
    df['tokens'] = df['title'].apply(word_tokenize)
    return df

def remove_links(tokens):
    return [w for w in tokens if not w.startswith('http')]

def remove_short_tokens(tokens, min_len=2):
    return [w for w in tokens if len(w) >= min_len]

def remove_stopwords(df):
    # Removes common filler words like "the", "and", "is", etc.
    stop_words = set(stopwords.words('english'))
    df['tokens'] = df['tokens'].apply(lambda tokens: [w for w in tokens if w not in stop_words])
    return df


def build_vocab(df, min_freq=5):
    # Builds a dictionary of words → index, only keeping those that appear at least `min_freq` times
    # Also returns raw word counts for inspection
    all_tokens = [word for tokens in df['tokens'] for word in tokens]
    word_counts = Counter(all_tokens)
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    return vocab, word_counts


def load_embeddings(filepath, max_vocab=50000):
    """
    Load pre-trained word embeddings from a .txt file.
    Returns: dict word → vector (as list of floats)
    """
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue  # skip header if present
            parts = line.rstrip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
            if len(embeddings) >= max_vocab:
                break
    return embeddings

def title_to_vec(tokens, embeddings, dim=100):
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if not vectors:
        return [0.0] * dim
    return list(np.mean(vectors, axis=0))

def vectorize_tokens(df, embeddings, vector_size=100):
    """
    Averages word embeddings for tokens in each title.
    Returns new DataFrame column: 'title_vector'
    """
    def avg_vector(tokens):
        vecs = [embeddings[word] for word in tokens if word in embeddings]
        if vecs:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(vector_size)

    df['title_vector'] = df['tokens'].apply(avg_vector)
    return df

def average_embedding(tokens, embeddings, dim=100):
    import numpy as np
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def load_embeddings(path):
    """
    Loads word embeddings from a .npy or .pkl file.
    """
    return np.load(path, allow_pickle=True).item()