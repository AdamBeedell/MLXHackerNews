import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from M.utils import load_embeddings
from collections import Counter
import ast

df = pd.read_csv("M/data/processed_hn_data.csv")

# ===== Prepare Columns =====
df['high_score'] = df['score'] > 100
df['tokens'] = df['tokens'].apply(ast.literal_eval)  # Converts string back to list

# ===== Create Subsets =====
high = df[df['high_score']]
low = df[~df['high_score']]

embedding_path = "enwiki_20180420_100d.txt"
embeddings = load_embeddings(embedding_path)

# ===== Count Word Frequencies in Each Subset =====
high_counts = Counter([word for tokens in high['tokens'] for word in tokens])
low_counts = Counter([word for tokens in low['tokens'] for word in tokens])


# ===== Compare Word Frequencies =====
word_scores = {}
for word in set(high_counts.keys()).union(low_counts.keys()):
    high_freq = high_counts[word]
    low_freq = low_counts[word]
    total = high_freq + low_freq
    if total >= 10:  # Only consider words that appear at least 10 times
        word_scores[word] = high_freq / total

# ===== Sort and Display Top Skewed Words =====
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

# ===== Histogram of Upvotes (Log Scale) =====
plt.figure(figsize=(8, 4))
plt.hist(df['score'], bins=100, log=True, color='skyblue')
plt.title("Histogram of Upvotes (Log Scale)")
plt.xlabel("Score")
plt.ylabel("Log Count")
plt.tight_layout()
plt.show()

# ===== Box Plot of Scores =====
plt.figure(figsize=(8, 2))
sns.boxplot(x=df['score'], color='lightgreen')
plt.title("Box Plot of Upvote Scores")
plt.tight_layout()
plt.show()

# ===== Title Length Histogram =====
plt.figure(figsize=(8, 4))
plt.hist(df['title_length'], bins=50, color='orange')
plt.title("Title Length Distribution")
plt.xlabel("Title Length (Characters)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ===== Scatter Plot: Title Length vs Score =====
plt.figure(figsize=(8, 5))
sns.scatterplot(x='title_length', y='score', data=df, alpha=0.5)
plt.title("Title Length vs. Upvote Score")
plt.xlabel("Title Length")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# ===== Top Authors by Total Score =====
top_authors_total = (
    df.groupby("by")["score"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n👑 Top 10 Authors by Total Score:")
print(top_authors_total)

# ===== Top Authors by Average Score (min 5 posts) =====
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

