# Defines your neural network architecture.

import torch.nn as nn

# === Hyperparameters ===
embedding_dim = 128
window_size = 2
vocab_size = 500 
batch_size = 64
epochs = 1
learning_rate = 0.001
max_words = None
# =======================

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        embed = self.embeddings(center_words)
        out = self.linear(embed)
        return out
    

class Regressor(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )