import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import torch
import pickle
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib


# Load embeddings (tensor)
embedding_matrix = torch.load("text8_embeddings.pt").cpu().numpy()

# Load vocabulary mapping word -> index
with open("text8_word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# Define embedding dimension dynamically
embedding_dim = embedding_matrix.shape[1]

# ===== Load Preprocessed Data =====
df = pd.read_csv("../data/processed_hn_data.csv")
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# ===== Extra Features =====
df['title_length'] = df['title'].str.len()
df['avg_word_length'] = df['title'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)

df['time'] = pd.to_datetime(df['time'])
df['day_of_week'] = df['time'].dt.dayofweek
df['hour_of_day'] = df['time'].dt.hour

time_feats = pd.get_dummies(df[['day_of_week', 'hour_of_day']].astype(str))

top_authors = df['by'].value_counts().head(50).index
df['author_feat'] = df['by'].apply(lambda x: x if x in top_authors else "other")
author_ohe = pd.get_dummies(df['author_feat'], prefix='author')

# ===== Upvote Label Bucketing =====
def label_score(score):
    if score < 30:
        return 0  # Low
    elif score < 100:
        return 1  # Medium
    else:
        return 2  # High

df['label'] = df['score'].apply(label_score)

# ===== Convert Titles to Embeddings =====
def tokens_to_avg_vec(tokens):
    vecs = [embedding_matrix[word_to_index[w]] for w in tokens if w in word_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(embedding_dim)

X_embed = np.vstack(df['tokens'].apply(tokens_to_avg_vec))

# ===== Combine All Features =====
X_extra = np.hstack([
    df[['title_length', 'avg_word_length']].values,
    time_feats.values,
    author_ohe.values
])

time_info = {col: time_feats[col].values[0:1] for col in time_feats.columns}
author_map = {col.split("_")[-1]: author_ohe[col].values[0:1] for col in author_ohe.columns}


# Combine embeddings and extra features
X = np.hstack([X_embed, X_extra])
y = df['label'].values

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Train Classifier =====
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ===== Evaluate =====
y_pred = clf.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== Predict Custom Title =====
def predict_upvotes(title, clf, word_to_index, embedding_matrix, author_map, time_info):
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import numpy as np

    stop_words = set(stopwords.words('english'))
    title_clean = re.sub(r'[^\w\s]', '', title.lower())
    tokens = [w for w in word_tokenize(title_clean) if w not in stop_words]

    # Average embedding
    vecs = [embedding_matrix[word_to_index[w]] for w in tokens if w in word_to_index]
    if not vecs:
        print("❗ Title contains unknown words only.")
        return None
    avg_vec = np.mean(vecs, axis=0)

    # Extra features
    title_length = len(title)
    avg_word_len = np.mean([len(w) for w in title.split()]) if title.split() else 0

    # Get the correct feature dimensions
    time_dim = len(next(iter(time_info.values())))
    author_dim = len(next(iter(author_map.values())))

    # Assume dummy values for author + time (or pass them in)
    day = "0"   # Monday
    hour = "12" # Noon

    time_vector = time_info.get(f"{day}_{hour}", np.zeros(time_dim))
    author_vector = author_map.get("other", np.zeros(author_dim))

    extra = np.concatenate([[title_length, avg_word_len], time_vector, author_vector])
    full_input = np.concatenate([avg_vec, extra]).reshape(1, -1)

    pred_class = clf.predict(full_input)[0]
    return ["Low", "Medium", "High"][pred_class]


# Now call the function:
title = "Ask HN: Should I learn Rust or Go?"
pred = predict_upvotes(title, clf, word_to_index, embedding_matrix, author_map, time_info)
if pred:
    print(f"\n🔮 Predicted upvote class for: \"{title}\" → {pred}")


# 🔍 Try custom title
pred = predict_upvotes(title, clf, word_to_index, embedding_matrix)
title = "Ask HN: Should I learn Rust or Go?"
if pred:
    print(f"\n🔮 Predicted upvote class for: \"{title}\" → {pred}")


import torch.nn.functional as F

def get_title_embedding(title, word_to_index, embeddings):
    words = title.lower().split()  # simple tokenization
    vectors = []
    for w in words:
        idx = word_to_index.get(w)
        if idx is not None:
            vectors.append(embeddings[idx])
    if vectors:
        return torch.stack(vectors).mean(dim=0)  # average pooling
    else:
        return torch.zeros(embeddings.size(1))  # zero vector if no words found


from sklearn.linear_model import LinearRegression
import numpy as np

hn_data = zip(df['title'], df['score'])

X = []  # list of pooled embeddings (numpy arrays)
y = []  # list of upvote scores

for title, score in hn_data:
    emb = get_title_embedding(title, word_to_index, embeddings).numpy()
    X.append(emb)
    y.append(score)

reg = LinearRegression()
reg.fit(X, y)


joblib.dump(reg, "linear_regression_model.joblib")




# ===== Intrinsic Evaluation (Word Similarity) =====
# def most_similar(word, embeddings, word_to_index, top_n=10):
#     if word not in word_to_index:
#         print(f"'{word}' not in vocabulary.")
#         return []
    
#     idx = word_to_index[word]
#     vec = embeddings[idx].reshape(1, -1)
#     similarities = cosine_similarity(vec, embeddings)[0]
#     sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]
#     idx_to_word = {i: w for w, i in word_to_index.items()}
#     return [(idx_to_word[i], similarities[i]) for i in sorted_indices]

# print("\n🔍 Most similar to 'python':")
# for word, score in most_similar("python", embedding_matrix, word_to_index):
#     print(f"{word:>12} → {score:.3f}")
