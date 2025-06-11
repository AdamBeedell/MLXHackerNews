from collections import Counter

MIN_COUNT = 5
TEXT8_PATH   = 'text8.txt'   # make sure this file is in your cwd

def get_vocab():
  counts = Counter()
  with open(TEXT8_PATH, 'r') as f:
      for line in f:
          counts.update(line.strip().split())

  vocab = [w for w,c in counts.items() if c >= MIN_COUNT] + ['<unk>']
  vocab_len = len(vocab)
  word2idx = {w:i for i,w in enumerate(vocab)}
  unk_idx   = word2idx['<unk>']

  return { "vocab": vocab, "vocab_len": vocab_len, "word2idx": word2idx, "unk_idx": unk_idx }
