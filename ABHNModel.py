import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
from urllib.parse import urlparse
from torch.utils.data import Dataset, DataLoader

# === Config ===
CSV_PATH = "cleantrainingdata.csv"
MODEL_PATH = "ABembeddingsfullmodel.pth"
BATCH_SIZE = 64
EMBED_DIM = 111

# === Load pre-trained embeddings model ===
embedding_model = torch.load(MODEL_PATH)
embedding_weights = embedding_model.emb.weight.detach().clone()

# === Load and index Hacker News data ===
df = pd.read_csv(CSV_PATH)

# Index usernames and domains
user2idx = {u: i for i, u in enumerate(df['by'].unique())}
domain2idx = {d: i for i, d in enumerate(df['domain'].unique())}

# Save for reuse
with open("user2idx.json", "w") as f:
    json.dump(user2idx, f)
with open("domain2idx.json", "w") as f:
    json.dump(domain2idx, f)

# === Tokenizer for titles ===
def tokenize(title):
    return title.lower().split()

# === Dataset ===
class HackerNewsDataset(Dataset):
    def __init__(self, df, word2idx, unk_idx):
        self.df = df
        self.word2idx = word2idx
        self.unk = unk_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = tokenize(row['title'])
        token_ids = [self.word2idx.get(w, self.unk) for w in tokens]
        context = torch.tensor(token_ids[:10])  # truncate/pad elsewhere later if needed

        user = user2idx.get(row['by'], 0)
        domain = domain2idx.get(row['domain'], 0)
        score = torch.tensor(row['score'], dtype=torch.float32)

        return context, user, domain, score

# === Create vocab from embedding model ===
idx2word = {i: w for w, i in embedding_model.emb.weight_to_index.items()} if hasattr(embedding_model.emb, 'weight_to_index') else None
word2idx = {w: i for i, w in enumerate(idx2word.values())} if idx2word else {w: i for i, w in enumerate(embedding_model.emb.weight.shape[0])}
unk_idx = word2idx.get('<unk>', 0)

# === DataLoader ===
dataset = HackerNewsDataset(df, word2idx, unk_idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
class HackerNewsRegressor(nn.Module):
    def __init__(self, emb_weights, num_users, num_domains, embed_dim):
        super().__init__()
        vocab_size = emb_weights.shape[0]

        self.word_emb = nn.Embedding.from_pretrained(emb_weights)
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.domain_emb = nn.Embedding(num_domains, embed_dim)

        self.fc1 = nn.Linear(embed_dim * 3, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, title, user, domain):
        title_vec = self.word_emb(title).mean(dim=1)
        user_vec = self.user_emb(user)
        domain_vec = self.domain_emb(domain)

        x = torch.cat([title_vec, user_vec, domain_vec], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)

# === Training loop placeholder ===
model = HackerNewsRegressor(
    emb_weights=embedding_weights,
    num_users=len(user2idx),
    num_domains=len(domain2idx),
    embed_dim=EMBED_DIM
)

print("Model ready. Add training loop next.")