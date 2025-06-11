import torch.nn as nn

# === 3) CBOW Model ===
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lin = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, 4) → average → (batch_size, embed_dim)
        v = self.emb(x).mean(dim=1)
        return nn.functional.log_softmax(self.lin(v), dim=1)
