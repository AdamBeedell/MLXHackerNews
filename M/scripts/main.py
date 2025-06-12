from M.utils import (
    load_hn_data,
    clean_titles,
    extract_time_features,
    tokenize_titles,
    remove_stopwords,
    remove_links,
    remove_short_tokens,
    load_embeddings,
    vectorize_tokens,
)

# Load processed data
df = load_hn_data(limit=10000, min_score=10)

# Clean & preprocess
df = clean_titles(df)
df = extract_time_features(df)
df = tokenize_titles(df)
df['tokens'] = df['tokens'].apply(remove_links)
df['tokens'] = df['tokens'].apply(remove_short_tokens)
df = remove_stopwords(df)

# Load pre-trained word embeddings (make sure the file path is correct)
embedding_path = "enwiki_20180420_100d.txt"
embeddings = load_embeddings(embedding_path)

# 👉 Convert each title's tokens to an average embedding vector
df = vectorize_tokens(df, embeddings)

# Optional: Show a sample vector
print("✅ Sample title vector:", df['title_vector'].iloc[0])