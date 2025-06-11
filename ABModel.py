## Adam's model .py

###add dependancies
import os
import bz2
import csv
import torch
print(torch.__version__)
print(torch.cuda.is_available())  ## looking for True
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import itertools

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


### hyperparameters
windowsize = 2  # words either side of the target word
windowsize = windowsize * 2 + 1 
split_ratio = 0.8  # 80% for training, 20% for testing
embed_dim = 111


### Goal - Import text8

text8 = bz2.open('wikipedia_data.txt.bz2', 'rt').read()  # Read the text8 dataset from a bz2 compressed file   #### Not actually .bz2 at the moment, but this is how it will be in the future
text8 = text8.split()  # Split the text into words
text8.append('<unk>')  # Add an unknown token to the vocabulary

#>>> len(text8)
#17005207
#>>> len(set(text8))
#253854
#
# print(f"Number of words in text8: {len(text8)}")  # Uncomment to see the number of words in the dataset
# print(f"First 10 words in text8: {text8[:10]}")  # Uncomment to see the first 10 words in the dataset
# print(f"Distinct words in text8: {len(set(text8))}")  # Uncomment to see the number of distinct words in the dataset")


### tokenize text8

vocablist = set(text8)  ## deduping, not sure this is required
vocabsize = len(vocablist)  # Number of unique words in the vocabulary
word2idx = {w: i for i, w in enumerate(sorted(vocablist))} ## i sets an index, w is the word

unk_idx = word2idx['<unk>']  # Index for the unknown token
idx2word = {i: w for w, i in word2idx.items()}

windows = list(zip(*[iter(text8)]*windowsize))  # Group words into batches of size batch_size

#3401041

split = int(len(windows) * split_ratio)  # Split the dataset into training and testing sets
train_windows = windows[:split]
test_windows = windows[split:]


#train_dataset = text8[:len(text8)*0.8]  # 80% for training
#test_dataset = text8[len(text8)*0.8:]  # 20% for testing

#def train_generator(windows, word2idx, unk_idx):
#    """Generator function to yield context and target pairs for training."""
#    for w1, w2, w3, w4, w5 in windows:
#        ctx = [word2idx.get(w, unk_idx) for w in (w1, w2, w4, w5)]
#        yield torch.tensor(ctx), tgt
#        tgt = word2idx.get(w3, unk_idx)
    

#traintensors = train_generator(train_dataset, word2idx, unk_idx)
#testtensors = train_generator(test_dataset, word2idx, unk_idx)



class MaskedCBOWDataset(torch.utils.data.IterableDataset):
    def __init__(self, windows, word2idx, unk_idx):
        self.windows = windows
        self.word2idx = word2idx
        self.unk_idx = unk_idx

    def __iter__(self):
        for w1, w2, w3, w4, w5 in self.windows:
            ctx = [self.word2idx.get(w, self.unk_idx) for w in (w1, w2, w4, w5)]
            tgt = self.word2idx.get(w3, self.unk_idx)
            yield torch.tensor(ctx), tgt


#train_dataset = MaskedCBOWDataset(train_windows, word2idx, unk_idx)
#train_loader = DataLoader(train_dataset, batch_size=128)

# Example usage of the generator

#for context, target in gen:
#    print(context, target)

#/eg


### create model architecture

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    MaskedCBOWDataset(train_windows, word2idx, unk_idx),
    batch_size=64,
    #shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    MaskedCBOWDataset(test_windows, word2idx, unk_idx),
    batch_size=64,
    #shuffle=False
)
#print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

for i, (context, target) in enumerate(train_loader):
    print(f"Batch {i}:")
    print(f"  Context shape: {context.shape}")  # expect [batch_size, 4]
    print(f"  Target shape:  {target.shape}")   # expect [batch_size]
    print(f"  First row:     {context[0].tolist()} → {target[0].item()}")
    if i == 2: break  # only show a few batches


### create model

class word2vec(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.emb = NN.Embedding(vocabsize, embed_dim)  # 111 to be different
        self.out = NN.Linear(embed_dim, vocabsize)     # predict vocab word from averaged context
    def forward(self, x):  # x: [batch, 4]
        x = self.emb(x)           # → [batch, 4, embed_dim]
        x = x.mean(dim=1)         # → [batch, embed_dim]  ← averaging context vectors
        x = F.relu(x)             # optional, but can help
        x = self.out(x)           # → [batch, vocab_size]
        return x                  # raw logits

loss_function = NN.CrossEntropyLoss()  # using built-in loss function
model = word2vec().to(device) ##create the model as described above
optimizer = optim.Adam(model.parameters(), lr=0.001) ### lr = learning rate, 0.001 is apparently a "normal" value. Adam is the optimizer chosen, also fairly default


##### do training

num_epochs = 1 ## passes through the dataset

for epoch in range(num_epochs):
    for context, target in train_loader: #note uses batches defined earlier
        
        context = context.to(device)  # move data to the selected device
        target = target.to(device)    # move data to the selected device
        optimizer.zero_grad() ### reset gradients each time

        outputs = model(context) # forward pass
        loss = loss_function(outputs, target)

        loss.backward() ## backprop method created by pytorch crossentropyloss function, very convenient
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



### / training


### train model  