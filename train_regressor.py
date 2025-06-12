import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import os
from train_word2vec_wiki import build_vocab, CBOWModel
import os
from sqlalchemy import create_engine

from tqdm import tqdm

import wandb

# Non-interactive W&B login using environment variable
api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)


# ----- Config -----
# Paths (update as needed)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://myuser:mypass@localhost:5432/mydb")
EMBEDDINGS_PATH = os.environ.get(
    "EMBEDDINGS_PATH",
    os.path.expanduser('~/Downloads/wiki_cbow_text8.pt')
)
WORD2IDX_PATH = os.environ.get(
    "WORD2IDX_PATH",
    os.path.expanduser('~/Downloads/wiki_cbow_text8_word2idx.pt')
)
EMBED_DIM = 200  # Must match your word2vec training!

# ----- Load Embeddings -----
word2idx = torch.load(WORD2IDX_PATH)
embed_weights = torch.load(EMBEDDINGS_PATH)
vocab_size = len(word2idx)

# Rebuild embedding layer for inference
embedding = nn.Embedding(vocab_size, EMBED_DIM)
embedding.weight.data.copy_(embed_weights)

# ----- Dataset -----
class HNTitleDataset(Dataset):
    def __init__(self, data_source, word2idx):
        if isinstance(data_source, pd.DataFrame):
            df = data_source
        else:
            df = pd.read_csv(data_source)
        self.titles = df['title'].astype(str).tolist()
        self.scores = df['score'].values.astype(np.float32)
        self.word2idx = word2idx

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        tokens = self.titles[idx].lower().split()
        idxs = [self.word2idx.get(t, 0) for t in tokens if t in self.word2idx]  # 0 is for OOV/pad
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(self.scores[idx], dtype=torch.float32)

def collate_batch(batch):
    idxs, scores = zip(*batch)
    # For each title, get average-pooled embedding (no padding needed since we just average)
    pooled = []
    for idx_seq in idxs:
        if len(idx_seq) == 0:
            pooled.append(torch.zeros(EMBED_DIM))  # handle empty titles
        else:
            pooled.append(embedding(idx_seq).mean(dim=0).detach())
    return torch.stack(pooled), torch.tensor(scores)

# ----- Model -----
class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----- Training Loop -----
def train():
    # Initialize Weights & Biases run
    wandb.init(
        project="hn-upvotes-regression",
        config={
            "embed_dim": EMBED_DIM,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "epochs": 20
        }
    )
    config = wandb.config
    print(f"[STEP] W&B initialized with config: {config}")

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STEP] Using device: {device}")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set to load data from Postgres")
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        """SELECT p.id, i.title, i.score::INTEGER AS score, p.time::TIMESTAMP AS time
            FROM hacker_news.items_by_year_2024 AS p
            JOIN hacker_news.items AS i USING (id)
            WHERE i.score IS NOT NULL
        """, engine
    )
    ds = HNTitleDataset(df, word2idx)
    print(f"[STEP] Dataset prepared: {len(ds)} examples")
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )
    print(f"[STEP] DataLoader ready with batch size {loader.batch_size}")

    # Instantiate model and move it to device
    model = Regressor(EMBED_DIM)
    model.to(device)
    # Watch model parameters and gradients
    wandb.watch(model, log="all", log_freq=10)
    print(f"[STEP] Model instantiated: {model}")
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3) #fixed learning rate
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Step‐decay learning‐rate scheduler: halve LR every 2 epochs
    scheduler = StepLR(
        optimizer,
        step_size=2,
        gamma=0.5
    )

    for epoch in range(config.epochs):
        print(f"[STEP] Starting epoch {epoch+1}/{config.epochs}")
        model.train()
        losses = []
        for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            # Transfer batch to device
            x, y = x.to(device), y.to(device)
            # Log progress every 500 batches
            if batch_idx and batch_idx % 500 == 0:
                print(f"[STEP] Epoch {epoch+1}, batch {batch_idx}/{len(loader)}")
                wandb.log({"batch": batch_idx})

            pred = model(x)
            y_log = torch.log1p(y) # log-transform the target, which is common for regression tasks with skewed targets
            loss = criterion(pred, y_log) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")
        # Log training loss to W&B
        print(f"[STEP] Logging epoch {epoch+1} loss to W&B")
        wandb.log({"epoch": epoch+1, "train_loss": np.mean(losses)})
        # Advance scheduler and log new learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[STEP] Adjusted learning rate to {current_lr:.6f}")
        wandb.log({"lr": current_lr})
    # Save model
    torch.save(model.state_dict(), "regressor.pt")
    # Log the trained model as a W&B artifact
    artifact = wandb.Artifact("hn-upvotes-regressor", type="model")
    artifact.add_file("regressor.pt")
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    train()