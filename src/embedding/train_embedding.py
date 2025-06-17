from operator import itemgetter
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import wandb

from cbow_text8_dataset import CBOWText8Dataset
from cbow_model import CBOW

from vocab import get_vocab
from utils import get_device

# === Hyperparameters ===
TEXT8_PATH = 'text8.txt'
EMBEDDING_DIM = 100
CONTEXT_SIZE  = 2            # 2 left + 2 right = 5-word window
BATCH_SIZE = 128
EPOCHS = 5
NUM_WORKERS = 4
LEARNING_RATE = 0.01

# === Device setup (use MPS on Apple Silicon if possible) ===
device = get_device()
vocab, vocab_len, word2idx, unk_idx = itemgetter("vocab", "vocab_len", "word2idx", "unk_idx")(get_vocab())

def collate(batch):
    contexts, targets = zip(*batch)
    return torch.stack(contexts), torch.tensor(targets, dtype=torch.long)

def main():
    # Start a new wandb run to track this script.
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="attp-ml-institute",
        # Set the wandb project where this run will be logged.
        project="cbow-word2vec",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "CBOW",
            "dataset": "text8",
            "epochs": EPOCHS,
        },
    )

    # Dataset & DataLoader
    dataset = CBOWText8Dataset(TEXT8_PATH, word2idx, context_size=CONTEXT_SIZE, device=device)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate
    )

    # Model, loss, optimizer
    model = CBOW(vocab_len, EMBEDDING_DIM).to(device)

    # log the model
    wandb.watch(model, log="all", log_freq=100)

    loss_fn = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Run epochs
    for epoch in range(1, EPOCHS+1):
        print(f"Starting Epoch {epoch}")
        total_loss = 0.0

        for contexts, targets in tqdm(
            loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch"
        ):
            # move data to the selected device
            contexts = contexts.to(device)
            targets  = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(contexts)
            loss = loss_fn(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS}  Loss: {total_loss:.2f}")

    # 8. Save model weights locally
    model_path = "cbow-word2vec-model.pth"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(
        name="cbow-word2vec",
        type="model",
        description="CBOW Word2Vec algorithm trained on "
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()
    