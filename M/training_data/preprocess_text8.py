# Preprocessing specifically for Hacker News data.

def get_training_pairs():
    from datasets import load_dataset
    from collections import Counter

    # Load data
    dataset = load_dataset("afmck/text8")
    full_text = dataset['train'][0]['text']
    words = full_text.split()[:100_000]  # trim for speed

    # Build vocab
    counter = Counter(words)
    top_words = counter.most_common(500)
    word_to_index = {'<UNK>': 0}
    for idx, (word, _) in enumerate(top_words):
        word_to_index[word] = idx + 1

    # Convert words to ids
    data = [word_to_index.get(word, 0) for word in words]

    # Generate (center, context) pairs
    def generate_pairs(data, window=2):
        pairs = []
        for i in range(window, len(data) - window):
            center = data[i]
            context = data[i - window:i] + data[i+1:i+window+1]
            for ctx in context:
                pairs.append((center, ctx))
        return pairs

    return generate_pairs(data), word_to_index