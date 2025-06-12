# train_hn_embeddings.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast

from training_data import get_training_pairs, get_loader
from model import SkipGramModel

# ===== Hyperparameters =====
embedding_dim = 128
window_size = 2
vocab_size = 500
batch_size = 64
epochs = 20
learning_rate = 0.001
max_words = None

# ===== Load Hacker News Tokens =====
df = pd.read_csv("../data/processed_hn_data.csv")
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# Flatten tokens into a single list of words
words = [word for tokens in df['tokens'] for word in tokens]

# ===== Generate Training Data =====
pairs, word_to_index = get_training_pairs(
    words, 
    window_size=window_size, 
    vocab_size=vocab_size, 
    max_words=max_words
)
loader = get_loader(pairs, batch_size=batch_size)

# ===== Define Model, Loss, Optimizer =====
model = SkipGramModel(vocab_size=len(word_to_index), embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===== Training Loop =====
for epoch in range(epochs):
    total_loss = 0
    for center, context in loader:
        optimizer.zero_grad()
        output = model(center)
        loss = criterion(output, context)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ===== Save Model & Vocab =====
torch.save(model.state_dict(), "hn_embeddings.pt")
import pickle
with open("hn_word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

print("\n✅ Finished training Hacker News embeddings.")

