from operator import itemgetter
import re
import multiprocessing as mp

import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from embedding.cbow_model import CBOW
from database import fetch_hacker_news_info, fetch_hackernews_length
from vocab import get_vocab

EMBEDDING_DIM = 100
CONTEXT_SIZE  = 2 # 2 left + 2 right = 5-word window
BATCH_SIZE = 256
EPOCHS = 5
NUM_WORKERS = 4

vocab, vocab_len, word2idx, unk_idx = itemgetter("vocab", "vocab_len", "word2idx", "unk_idx")(get_vocab())

# === Device setup (use MPS on Apple Silicon if possible) ===
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

cbow_model = CBOW(vocab_len, EMBEDDING_DIM).to(device)
cbow_model.load_state_dict(torch.load('cbow-word2vec-model.pth'))
cbow_model.eval()

# Function to get word embeddings
def get_word_embedding(word, model, word2idx, unk_idx, device):
    """
    Retrieves the embedding for a single word.

    Args:
        word (str): The word to get the embedding for.
        model (CBOW): The loaded CBOW model.
        word2idx (dict): The word-to-index mapping.
        unk_idx (int): The index for unknown words.
        device (torch.device): The device to perform inference on.

    Returns:
        torch.Tensor: The embedding vector for the word, or None if the word is not in vocab.
    """
    idx = word2idx.get(word, unk_idx)
    # For a true word embedding, we typically use the `emb` layer directly.
    # The CBOW model's `forward` method averages context words,
    # but for individual word embeddings, we want the lookup.
    with torch.no_grad(): # No need to calculate gradients for inference
        # If you want the embedding of the target word itself, not its context average.
        # The embedding layer directly holds the word vectors.
        # We pass a single index, so we need to unsqueeze it to make it a batch of 1.
        embedding = model.emb(torch.tensor([idx], dtype=torch.long).to(device)).squeeze(0)
    return embedding

# === Function to get text embeddings (from previous inference.py) ===
def process_text_for_embeddings(text, model, word2idx, unk_idx, device, aggregation_method='mean'):
    words = re.findall(r'\b\w+\b', text.lower())
    
    word_embeddings = []
    with torch.no_grad():
        for word in words:
            idx = word2idx.get(word, unk_idx)
            embedding = model.emb(torch.tensor([idx], dtype=torch.long).to(device))
            word_embeddings.append(embedding)

    if not word_embeddings:
        return torch.zeros(EMBEDDING_DIM).to(device)

    stacked_embeddings = torch.cat(word_embeddings, dim=0)

    if aggregation_method == 'mean':
        text_embedding = torch.mean(stacked_embeddings, dim=0)
    elif aggregation_method == 'sum':
        text_embedding = torch.sum(stacked_embeddings, dim=0)
    else:
        raise ValueError("Invalid aggregation_method. Choose 'mean' or 'sum'.")

    return text_embedding
  
def fetch_hacker_news_data(limit = 10000, offset = 0, include_comments = False):
    return [post for post in fetch_hacker_news_info(limit, offset, include_comments)]
class HackerNewsIterableDataset(IterableDataset):
    def __init__(self, fetch_fn, batch_size=128,
                 include_comments=False,
                 cbow_model=None, word2idx=None, unk_idx=None, device=None,
                 start=0, end=None):
        self.fetch_fn = fetch_fn
        self.batch_size = batch_size
        self.include_comments = include_comments
        self.cbow   = cbow_model
        self.w2i    = word2idx
        self.unk    = unk_idx
        self.device = device
        self.start  = start
        self.end    = end

    def __iter__(self):
        offset = self.start
        yielded = 0
        while True:
            raw_batch = self.fetch_fn(limit=self.batch_size,
                                      offset=offset,
                                      include_comments=self.include_comments)
            if not raw_batch:
                return
            for item in raw_batch:
                if self.end is not None and yielded >= (self.end - self.start):
                    return
                emb = process_text_for_embeddings(
                    item.title, self.cbow, self.w2i, self.unk,
                    self.device, aggregation_method='mean'
                )
                yield emb.to(self.device), torch.tensor([item.score], dtype=torch.float32).to(self.device)
                yielded += 1
            offset += self.batch_size

    def __len__(self):
        return fetch_hackernews_length()

class UpvotePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_predictor(model, dataloader, loss_fn, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in tqdm(dataloader, desc=f"Predictor Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Predictor Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

def main():
    
    print("Fetching Hacker News data...")
    # Create the regression dataset
    full_len = len(HackerNewsIterableDataset(fetch_hacker_news_data, batch_size=BATCH_SIZE, 
                                            include_comments=False,
                                            cbow_model=cbow_model, word2idx=word2idx,
                                            unk_idx=unk_idx, device=device))
    train_len  = int(0.8 * full_len)
    train_ds = HackerNewsIterableDataset(
        fetch_hacker_news_data, batch_size=BATCH_SIZE,
        include_comments=False,
        cbow_model=cbow_model, word2idx=word2idx, unk_idx=unk_idx, device=device,
        start=0, end=train_len
    )
    val_ds   = HackerNewsIterableDataset(
        fetch_hacker_news_data, batch_size=BATCH_SIZE,
        include_comments=False,
        cbow_model=cbow_model, word2idx=word2idx, unk_idx=unk_idx, device=device,
        start=train_len, end=full_len
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Part 3: Train the Upvote Predictor Model ---
    INPUT_DIM = EMBEDDING_DIM # The size of the aggregated comment embedding
    HIDDEN_DIM = 64 # A configurable hidden layer size
    OUTPUT_DIM = 1  # Predicting a single continuous value (upvotes)

    predictor_model = UpvotePredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    predictor_loss_fn = nn.MSELoss() # Mean Squared Error for regression
    predictor_optimizer = optim.Adam(predictor_model.parameters(), lr=0.001)

    print("\n--- Training Upvote Predictor ---")
    train_predictor(predictor_model, train_loader, predictor_loss_fn, predictor_optimizer, epochs=100, device=device)

    # --- Part 4: Evaluate and Predict ---
    predictor_model.eval() # Set predictor model to evaluation mode
    
    # Calculate performance on validation set
    val_losses = []
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = predictor_model(features)
            loss = predictor_loss_fn(predictions, labels)
            val_losses.append(loss.item())
            all_predictions.extend(predictions.cpu().squeeze().tolist())
            all_targets.extend(labels.cpu().squeeze().tolist())
    
    avg_val_loss = np.mean(val_losses)
    print(f"\nAverage Validation MSE: {avg_val_loss:.4f}")

    # Example of predicting for new data
    print("\n--- Example Predictions for New Articles ---")
    new_post = [
        {"title": "This is an amazing breakthrough in AI!"}, # Likely high upvotes
        {"title": "Just a quick thought."}, # Medium upvotes
        {"title": "Random comment."} # Low upvotes
    ]

    for i, post in enumerate(new_post):
        new_article_all_embeddings = []
        new_article_all_embeddings.append(process_text_for_embeddings(
            post.get("title"), cbow_model, word2idx, unk_idx, device, aggregation_method='mean'
        ))
        
        if new_article_all_embeddings:
            new_article_feature = torch.stack(new_article_all_embeddings).mean(dim=0).unsqueeze(0).to(device)
        else:
            new_article_feature = torch.zeros(1, EMBEDDING_DIM).to(device) # Batch size 1
        
        with torch.no_grad():
            predicted_upvotes = predictor_model(new_article_feature).item()
        
        print(f"Article {i+1} \"{post.get("title")}\" Predicted Upvotes = {predicted_upvotes:.2f}")
    
    model_path = "hn-predictor-model.pth"
    torch.save(predictor_model.state_dict(), model_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()