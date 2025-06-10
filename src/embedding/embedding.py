import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# === Hyperparameters ===
TEXT8_PATH   = 'text8.txt'   # make sure this file is in your cwd
EMBEDDING_DIM = 100
CONTEXT_SIZE  = 2            # 2 left + 2 right = 5-word window
MIN_COUNT = 5
BATCH_SIZE = 128
EPOCHS = 5
NUM_WORKERS = 4

# === Device setup (use MPS on Apple Silicon if possible) ===
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# === 1) Build vocab from text8 in one pass ===
counts = Counter()
with open(TEXT8_PATH, 'r') as f:
    for line in f:
        counts.update(line.strip().split())

vocab = [w for w,c in counts.items() if c >= MIN_COUNT] + ['<unk>']
word2idx = {w:i for i,w in enumerate(vocab)}
unk_idx   = word2idx['<unk>']

class CBOWText8Dataset(IterableDataset):
    def __init__(self, file_path, w2i, context_size=2):
        self.file_path = file_path
        self.w2i       = w2i
        self.unk       = w2i['<unk>']
        self.ctx       = context_size

    def __iter__(self):
        worker_info = get_worker_info()
        # open the huge file once per worker
        with open(self.file_path, 'r') as f:
            for chunk in f:                              # one giant line
                words = chunk.strip().split()
                length = len(words)
                window_idx = 0
                for i in range(self.ctx, length - self.ctx):
                    if worker_info and window_idx % worker_info.num_workers != worker_info.id:
                        window_idx += 1
                        continue

                    # build context + target
                    ctx_idxs = [
                        self.w2i.get(words[i+j], self.unk)
                        for j in range(-self.ctx, self.ctx+1) if j != 0
                    ]
                    tgt_idx  = self.w2i.get(words[i], self.unk)

                    yield torch.tensor(ctx_idxs, dtype=torch.long), tgt_idx
                    window_idx += 1

def collate(batch):
    contexts, targets = zip(*batch)
    return torch.stack(contexts), torch.tensor(targets, dtype=torch.long)

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

def main():
    # Dataset & DataLoader
    dataset = CBOWText8Dataset(TEXT8_PATH, word2idx, context_size=CONTEXT_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate
    )

    # Model, loss, optimizer
    model = CBOW(len(vocab), EMBEDDING_DIM).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Run epochs
    for epoch in range(1, EPOCHS+1):
        print(f"Starting Epoch {epoch}")
        total_loss = 0.0

        for contexts, targets in loader:
            # move data to the selected device
            contexts = contexts.to(device)
            targets  = targets.to(device)
            optimizer.zero_grad()
            log_probs = model(contexts)
            loss = loss_fn(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS}  Loss: {total_loss:.2f}")

if __name__ == "__main__":
    main()
    