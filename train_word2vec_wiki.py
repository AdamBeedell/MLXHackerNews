#!/usr/bin/env python3
"""
train_cbow_pytorch.py

Train a CBOW Word2Vec model with negative sampling on Wikipedia.
"""

import argparse, os, random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import bz2
import xml.etree.ElementTree as ET

def build_vocab(dump_path, min_count, max_pages=None):
    """
    1) Stream Wikipedia dump once to count tokens.
    2) Keep only words with frequency >= min_count.
    Returns word2idx dict.
    """
    # Detect BZ2 compression by magic header
    with open(dump_path, 'rb') as _f:
        header = _f.read(3)
    if header != b'BZh':
        counter = Counter()
        with open(dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.lower().split()
                counter.update(tokens)
        filtered_words = [w for w, c in counter.items() if c >= min_count]
        return {w: idx for idx, w in enumerate(filtered_words)}
    counter = Counter()
    # Stream and parse XML <text> elements
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('end',))
        page_count = 0
        for event, elem in context:
            if elem.tag.endswith('text'):
                page_count += 1
                if max_pages and page_count > max_pages:
                    break
                tokens = (elem.text or '').lower().split()
                counter.update(tokens)
                elem.clear()
    # Only keep words meeting frequency threshold
    filtered_words = [w for w, c in counter.items() if c >= min_count]
    # Assign new 0-based indices
    return {w: idx for idx, w in enumerate(filtered_words)}

class CBOWDataset(IterableDataset):
    """
    IterableDataset yielding:
      context_ids: LongTensor[2*window]
      target_id:   LongTensor[]
      neg_ids:     LongTensor[neg_samples]
    """
    def __init__(self, dump_path, word2idx, window, neg_samples, max_pages=None): #neg_samples is the number of negative samples per target word
        self.dump_path  = dump_path
        self.word2idx   = word2idx
        self.vocab_size = len(word2idx)
        self.window     = window
        self.neg_samples= neg_samples
        self.max_pages  = max_pages

    def __iter__(self):
        page_count = 0
        # Detect BZ2 compression by magic header
        with open(self.dump_path, 'rb') as _f:
            header = _f.read(3)
        if header != b'BZh':
            with open(self.dump_path, 'r', encoding='utf-8') as f:
                tokens = f.read().lower().split()
            ids = [self.word2idx[t] for t in tokens if t in self.word2idx]
            for i in range(self.window, len(ids) - self.window):
                context_ids = ids[i-self.window:i] + ids[i+1:i+1+self.window]
                target_id = ids[i]
                neg_ids = [random.randrange(self.vocab_size) for _ in range(self.neg_samples)]
                yield (
                    torch.tensor(context_ids, dtype=torch.long),
                    torch.tensor(target_id,    dtype=torch.long),
                    torch.tensor(neg_ids,      dtype=torch.long)
                )
            return
        # Stream <text> tags from the compressed dump
        with bz2.open(self.dump_path, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('end',))
            for event, elem in context:
                if elem.tag.endswith('text'):
                    page_count += 1
                    if self.max_pages and page_count > self.max_pages:
                        return
                    tokens = (elem.text or '').lower().split()
                    ids = [self.word2idx[t] for t in tokens if t in self.word2idx]
                    # Slide a window over each center position
                    for i in range(self.window, len(ids) - self.window):
                        context_ids = ids[i-self.window:i] + ids[i+1:i+1+self.window]
                        target_id   = ids[i]
                        neg_ids     = [random.randrange(self.vocab_size) for _ in range(self.neg_samples)]
                        #yield returns a 3-tuple of PyTorch tensors
                        yield (
                            torch.tensor(context_ids, dtype=torch.long),
                            torch.tensor(target_id,   dtype=torch.long),
                            torch.tensor(neg_ids,     dtype=torch.long)
                        )
                    elem.clear()

class CBOWModel(nn.Module):
    """
    Simple CBOW: average context embeddings --> predict target (and negatives).
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed  = nn.Embedding(vocab_size, embed_dim) #in_embed for context words
        self.out_embed = nn.Embedding(vocab_size, embed_dim) #out_embed for target (and negative) words

    def forward(self, context_ids, target_ids, neg_ids):
        # context_ids: (batch, 2*window)
        # target_ids:  (batch,)
        # neg_ids:     (batch, neg_samples)
        v_ctx = self.in_embed(context_ids).mean(dim=1)     # (batch, embed_dim)
        u_pos = self.out_embed(target_ids)                 # (batch, embed_dim)
        pos_score = (v_ctx * u_pos).sum(dim=1)             # dot product → (batch,)

        u_neg = self.out_embed(neg_ids)                    # (batch, neg_samples, embed_dim)
        neg_score = torch.bmm(u_neg, v_ctx.unsqueeze(2))   # (batch, neg, 1)
        neg_score = neg_score.squeeze(2)                   # (batch, neg)

        return pos_score, neg_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump",       type=str, required=True,
                   help="Path to enwiki-latest-pages-articles.xml.bz2")
    p.add_argument("--output",     type=str, required=True,
                   help="Where to save final embeddings (torch .pt)")
    p.add_argument("--embed-dim",  type=int, default=100)
    p.add_argument("--window",     type=int, default=2)
    p.add_argument("--min-count",  type=int, default=100)
    p.add_argument("--neg-samples",type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs",     type=int, default=1)
    p.add_argument("--lr",         type=float, default=0.05)
    p.add_argument(
        "--max-pages", type=int, default=None,
        help="Maximum number of Wikipedia pages to process (for debugging)"
    )
    p.add_argument("--no-save", action="store_true",
                   help="If set, skip saving the final embeddings to disk")
    args = p.parse_args()

    print("Building vocabulary…")
    word2idx = build_vocab(args.dump, args.min_count, args.max_pages)
    # Save vocabulary mapping separately
    vocab_path = os.path.splitext(args.output)[0] + '_word2idx.pt'
    torch.save(word2idx, vocab_path)
    print(f"Saved vocabulary mapping to {vocab_path}")
    print(f"Vocab size: {len(word2idx)} tokens")

    dataset = CBOWDataset(
        dump_path   = args.dump,
        word2idx    = word2idx,
        window      = args.window,
        neg_samples = args.neg_samples,
        max_pages   = args.max_pages
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = CBOWModel(len(word2idx), args.embed_dim).to(device)
    optimizer= torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for context, target, neg in loader:
            context, target, neg = context.to(device), target.to(device), neg.to(device)
            pos_score, neg_score = model(context, target, neg)
            # CBOW negative-sampling loss
            loss = -F.logsigmoid(pos_score).mean() \
                   -F.logsigmoid(-neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {total_loss:.4f}")

    if not args.no_save:
        embeds = model.in_embed.weight.data.cpu()
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(embeds, args.output)
        print(f"Saved embeddings to {args.output}")

if __name__=="__main__":
    main()

"""
python train_word2vec_wiki.py \
  --dump ~/data/wiki/enwiki-latest-pages-articles.xml.bz2 \
  --output tmp.pt \
  --embed-dim 50 \
  --window 2 \
  --min-count 300 \
  --neg-samples 3 \
  --batch-size 512 \
  --epochs 1 \
  --lr 0.1 \
  --max-pages 500 \
  --no-save

  python train_word2vec_wiki.py \
  --dump ~/data/wiki/enwiki-latest-pages-articles.xml.bz2 \
  --output ./wiki_cbow.pt \
  --embed-dim 50 \
  --window 2 \
  --min-count 300 \
  --neg-samples 3 \
  --batch-size 512 \
  --epochs 1 \
  --lr 0.1 \
  --max-pages 500
"""