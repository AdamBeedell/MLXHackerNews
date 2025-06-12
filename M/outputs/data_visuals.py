import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import numpy as np

# ===== Load Data =====
df = pd.read_csv("../data/processed_hn_data.csv")
df['high_score'] = df['score'] > 100
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# ===== Create Subsets =====
high = df[df['high_score']]
low = df[~df['high_score']]

# ===== Load Word Embeddings from .txt =====
embedding_path = "../data/enwiki_20180420_100d.txt"

def load_embeddings(path, max_vocab=50000):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue  # Skip header
            parts = line.strip().split()
            word = parts[0]
            vec = list(map(float, parts[1:]))
            embeddings[word] = vec
            if len(embeddings) >= max_vocab:
                break
    return embeddings

embeddings = load_embeddings(embedding_path)

# ===== Word Frequency Skew =====
high_counts = Counter([word for tokens in high['tokens'] for word in tokens])
low_counts = Counter([word for tokens in low['tokens'] for word in tokens])

word_scores = {}
for word in set(high_counts.keys()).union(low_counts.keys()):
    high_freq = high_counts[word]
    low_freq = low_counts[word]
    total = high_freq + low_freq
    if total >= 10:
        word_scores[word] = high_freq / total

popular_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("\n🔥 Words most skewed toward high-scoring posts:")
for word, score in popular_words:
    print(f"{word:<15} → {score:.2f}")

# ===== Bar Plot of Top Skewed Words =====
top_words = dict(popular_words[:100])
plt.figure(figsize=(10, 4))
sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
plt.xlabel("High Score Word Ratio")
plt.title("Top Words in High-Scoring Titles")
plt.tight_layout()
plt.show()

# ===== Time Features =====
df['time'] = pd.to_datetime(df['time'])
df['day_of_week'] = df['time'].dt.dayofweek
df['hour_of_day'] = df['time'].dt.hour

# ===== Heatmap: Mean Score by Day & Hour =====
pivot = df.pivot_table(index="hour_of_day", columns="day_of_week", values="score", aggfunc="mean")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, cmap="YlGnBu", annot=False)
plt.title("Average Upvotes by Hour and Day")
plt.xlabel("Day of Week (0 = Monday)")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.show()

# ===== Violin Plot of Score by Day =====
plt.figure(figsize=(10, 4))
sns.violinplot(x="day_of_week", y="score", data=df)
plt.title("Score Distribution by Day of Week")
plt.xlabel("Day of Week (0 = Monday)")
plt.tight_layout()
plt.show()

# ===== Bin-Based Box Plot: Title Length vs Score =====
if 'title_length' not in df.columns:
    df['title_length'] = df['title'].str.len()
df['length_bin'] = pd.cut(df['title_length'], bins=[0, 40, 60, 80, 100, 120, 200])

plt.figure(figsize=(10, 5))
sns.boxplot(x="length_bin", y="score", data=df)
plt.title("Score by Title Length (Binned)")
plt.xlabel("Title Length Bin")
plt.tight_layout()
plt.show()

# ===== Top Authors =====
top_authors_total = (
    df.groupby("by")["score"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n👑 Top 10 Authors by Total Score:")
print(top_authors_total)

author_counts = df['by'].value_counts()
frequent_authors = author_counts[author_counts >= 5].index

top_authors_avg = (
    df[df['by'].isin(frequent_authors)]
    .groupby("by")["score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\n📈 Top 10 Authors by Average Score (Min 5 posts):")
print(top_authors_avg)

# ===== Predicted vs Actual Plot (Replace with real data!) =====
y_test = np.array([10, 50, 100, 200, 300])
y_pred = np.array([12, 55, 90, 180, 310])

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Upvotes')
plt.ylabel('Predicted Upvotes')
plt.title('Predicted vs Actual Upvotes')
plt.grid(True)
plt.tight_layout()
plt.show()
