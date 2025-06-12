# HANDLES PREPARING TRAINING DATASET

from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Function to prepare SkipGram training pairs from a list of tokens
def get_training_pairs(words, window_size=2, vocab_size=500, max_words=100_000, unk_token='<UNK>'):
    if max_words:
        words = words[:max_words]

    # Count word frequencies and keep top vocab_size words
    counter = Counter(words)
    top_words = counter.most_common(vocab_size)

    # Build word-to-index mapping with <UNK> token for rare words
    word_to_index = {unk_token: 0}
    for idx, (word, _) in enumerate(top_words):
        word_to_index[word] = idx + 1

    # Convert words to their IDs (or 0 if unknown)
    data = [word_to_index.get(word, 0) for word in words]

    pairs = []
    # Generate SkipGram (center, context) pairs using the window
    for i in range(window_size, len(data) - window_size):
        context = data[i - window_size:i] + data[i + 1:i + window_size + 1]
        pairs.extend([(data[i], ctx) for ctx in context])

    return pairs, word_to_index

# PyTorch Dataset wrapping the pairs
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return center, context

# Create DataLoader to feed batches
def get_loader(pairs, batch_size=64):
    dataset = SkipGramDataset(pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Main block to test the pipeline
if __name__ == "__main__":
    dataset = load_dataset("afmck/text8")
    full_text = dataset['train'][0]['text']
    words = full_text.split()  # No extra cleaning needed for Text8

    pairs, word_to_index = get_training_pairs(words, max_words=100_000)
    loader = get_loader(pairs)

    print(f"Total pairs: {len(pairs)}")
    for center, context in loader:
        print(center, context)
        break  # Just show first batch
