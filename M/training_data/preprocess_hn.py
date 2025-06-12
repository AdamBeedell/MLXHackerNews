import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def load_hn_data(limit=100000, min_score=10):
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
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return df

def tokenize_titles(df):
    df['tokens'] = df['title'].apply(word_tokenize)
    return df

def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    df['tokens'] = df['tokens'].apply(lambda tokens: [w for w in tokens if w not in stop_words])
    return df


df = pd.read_csv("../data/processed_hn_data.csv")
print(len(df))
print(df['score'].describe())
