import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from training_data import get_training_pairs, get_loader
from model import SkipGramModel
import pickle

# Hyperparameters 
embedding_dim = 128
window_size = 2
vocab_size = 5000
batch_size = 64
epochs = 20
learning_rate = 0.001
max_words = None

# Load and tokenize text
dataset = load_dataset("afmck/text8")
full_text = dataset['train'][0]['text']
words = full_text.split()

# Prepare data
pairs, word_to_index = get_training_pairs(words, window_size=window_size, vocab_size=vocab_size, max_words=max_words)
loader = get_loader(pairs, batch_size=batch_size)

# Create model, loss, optimizer
model = SkipGramModel(vocab_size=len(word_to_index), embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
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
    print(f"Epoch {epoch+1}, Loss: {avg_loss}.4")

    # Save checkpoint after each epoch
    torch.save(model.state_dict(), f"skipgram_epoch{epoch+1}.pth")

# ===== Save final embeddings and vocab =====

# Save the final embeddings weights (tensor)
torch.save(model.embeddings.weight.data.cpu(), "text8_embeddings.pt")

# Save the word_to_index dictionary
with open("text8_word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

print("\n✅ Finished training text8 embeddings.")
