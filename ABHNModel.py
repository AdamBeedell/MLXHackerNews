import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import bz2
from urllib.parse import urlparse
from torch.utils.data import Dataset, DataLoader

# === Config ===
CSV_PATH = "cleantrainingdata.csv"
MODEL_PATH = "ABembeddingsfullmodel.pth"
BATCH_SIZE = 64
EMBED_DIM = 111
MAX_TITLE_LEN = 20 



# previous word2vec class
class word2vec(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.emb = NN.Embedding(vocabsize, embed_dim)  # 111 to be different
        self.out = NN.Linear(embed_dim, vocabsize)     # predict vocab word from averaged context
    def forward(self, x):  # x: [batch, 4]
        x = self.emb(x)           # → [batch, 4, embed_dim]
        x = x.mean(dim=1)         # → [batch, embed_dim]  ← averaging context vectors
        x = F.relu(x)             # optional, but can help
        x = self.out(x)           # → [batch, vocab_size]
        return x                  # raw logits



#device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available, otherwise CPU


# === Load pre-trained embeddings model ===
embedding_model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))  ### dont map to CPU on computa gpu
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

        title = row['title']
        tokens = title.split()  # or use a proper tokenizer if needed
        token_ids = [self.word2idx.get(w, self.unk) for w in tokens]
        token_ids = token_ids[:MAX_TITLE_LEN]  # truncate
        token_ids += [self.unk] * (MAX_TITLE_LEN - len(token_ids))  # pad

        context = torch.tensor(token_ids, dtype=torch.long)

        user = torch.tensor(user2idx.get(row['by'], 0), dtype=torch.long)
        domain = torch.tensor(domain2idx.get(row['domain'], 0), dtype=torch.long)
        score = torch.tensor(row['score'], dtype=torch.float32)

        return context, user, domain, score

# === Create vocab from embedding model ===
text8 = bz2.open('wikipedia_data.txt.bz2', 'rt').read()  # Read the text8 dataset from a bz2 compressed file   #### Not actually .bz2 at the moment, but this is how it will be in the future
text8 = text8.split()  # Split the text into words
text8.append('<unk>')  # Add an unknown token to the vocabulary

vocablist = set(text8)  ## deduping, not sure this is required
vocabsize = len(vocablist)  # Number of unique words in the vocabulary
word2idx = {w: i for i, w in enumerate(sorted(vocablist))} ## i sets an index, w is the word

unk_idx = word2idx['<unk>']  # Index for the unknown token
idx2word = {i: w for w, i in word2idx.items()}



#idx2word = {i: w for w, i in embedding_model.emb.weight_to_index.items()} if hasattr(embedding_model.emb, 'weight_to_index') else None
#word2idx = {w: i for i, w in enumerate(idx2word.values())} if idx2word else {w: i for i, w in enumerate(embedding_model.emb.weight.shape[0])}
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

# === Training Setup ===
loss_function = NN.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for title_vecs, domain_idxs, user_idxs, scores in dataloader:
        title_vecs = title_vecs.to(device)
        domain_idxs = domain_idxs.to(device)
        user_idxs = user_idxs.to(device)
        scores = scores.to(device).float()

        optimizer.zero_grad()
        predictions = model(title_vecs, domain_idxs, user_idxs)
        loss = loss_function(predictions, scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "ABHNModel.pth")