import torch
import matplotlib.pyplot as plt
import os
from train_word2vec_wiki import build_vocab, CBOWModel
from sklearn.manifold import TSNE

dump_path = os.path.expanduser("~/data/text8/text8.txt")
word2idx = build_vocab(
    dump_path=dump_path,
    min_count=300,
    max_pages=5000
)

model = CBOWModel(len(word2idx), embed_dim=100)

embed_file = os.path.expanduser("~/Downloads/embeddings.pt")
if not os.path.isfile(embed_file):
    raise FileNotFoundError(f"Embeddings file not found: {embed_file}. "
                            "Please rerun training without --no-save to generate it.")
model.in_embed.weight.data = torch.load(embed_file)

embeds = model.in_embed.weight.data.cpu()

# Build reverse mapping:
idx2word = {idx: w for w, idx in word2idx.items()}

# Nearest-neighbor function
def nearest(word, topn=5):
    if word not in word2idx:
        return []
    vecs = embeds  # (vocab, dim)
    target = embeds[word2idx[word]].unsqueeze(0)  # (1, dim)
    sims = torch.matmul(target, vecs.t())[0]     # (vocab,)
    best = sims.topk(topn+1).indices.tolist()    # includes the word itself
    # Exclude the query word and return topn
    return [idx2word[i] for i in best if i != word2idx[word]][:topn]

# Test a few sample words:
test_words = ["science", "computer", "data", "research", "model"]
print("Nearest neighbors by cosine similarity:")
for w in test_words:
    neighbors = nearest(w, topn=10)
    print(f"  {w:10s} → {neighbors}")

# Visualize the first 100 embeddings
plt.figure(figsize=(10, 10))
plt.imshow(embeds[:100].numpy())
plt.colorbar()
plt.title("First 100 word embeddings")
plt.show()

# Dimensionality reduction & scatter for a small sample of words
sample_size = min(200, embeds.shape[0])
sample_idxs = list(range(sample_size))
sample_embeds = embeds[sample_idxs].numpy()
sample_labels = [idx2word[i] for i in sample_idxs]

# Compute t-SNE
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
points = tsne.fit_transform(sample_embeds)

# Plot t-SNE result
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.7)
for i, label in enumerate(sample_labels):
    plt.text(points[i, 0], points[i, 1], label, fontsize=6)
plt.title("t-SNE projection of first 200 word embeddings")
plt.xticks([]); plt.yticks([])
plt.show()

"""
That plot is just a visual “heatmap” of your first 100 embedding vectors—each row is one word’s 50-dimensional embedding, 
and each column is one of those embedding dimensions. 
The color at (row r, col c) shows the numerical value of the r-th word’s c-th coordinate (blue for negative, yellow for positive).
"""