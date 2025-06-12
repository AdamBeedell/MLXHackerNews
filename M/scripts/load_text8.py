from datasets import load_dataset

from collections import Counter

# Step 1: Load text
dataset = load_dataset("afmck/text8")
full_text = dataset['train'][0]['text']
text = " ".join(full_text.split()[:100000])

# Step 2: Tokenize words
words = text.split()

# Step 3: Count frequencies
counter = Counter(words)
top_words = counter.most_common(500)

# Step 4: Assign IDs
word_to_index = {'<UNK>': 0}
for idx, (word, _) in enumerate(top_words):
    word_to_index[word] = idx + 1

# Step 5: Turn words into numbers
data = []
for word in words:
    index = word_to_index.get(word, 0)
    data.append(index)

# Step 6: Generate (center, context) pairs

def generate_skipgram_pairs(data, window_size):
    pairs = []

    for i in range(window_size, len(data) - window_size):
        center = data[i]
        context = data[i - window_size : i] + data[i + 1 : i + window_size + 1]
        
        for ctx_word in context:
            pairs.append((center, ctx_word))

    return pairs  # Always return the result from the function

# Example use:
window_size = 2
data = [10, 20, 30, 40, 50, 60]
pairs = generate_skipgram_pairs(data, window_size)
print(pairs[:10])
