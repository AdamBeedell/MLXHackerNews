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

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


### hyperparameters
windowsize = 2  # words either side of the target word
windowsize = windowsize * 2 + 1 


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
word2idx = {w: i for i, w in enumerate(sorted(vocablist))} ## i sets an index, w is the word

unk_idx = word2idx['<unk>']  # Index for the unknown token
idx2word = {i: w for w, i in word2idx.items()}

windows = list(zip(*[iter(text8)]*windowsize))  # Group words into batches of size batch_size

#3401041

train_dataset = windows[:len(windows)*0.8]  # 80% for training
test_dataset = windows[len(windows)*0.8:]  # 20% for testing
#train_dataset = text8[:len(text8)*0.8]  # 80% for training
#test_dataset = text8[len(text8)*0.8:]  # 20% for testing

def train_generator(windows, word2idx, unk_idx):
    """Generator function to yield context and target pairs for training."""
    for w1, w2, w3, w4, w5 in windows:
        ctx = [word2idx.get(w, unk_idx) for w in (w1, w2, w4, w5)]
        tgt = word2idx.get(w3, unk_idx)
        yield torch.tensor(ctx), tgt
    

traintensors = train_generator(train_dataset, word2idx, unk_idx)
testtensors = train_generator(test_dataset, word2idx, unk_idx)



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


train_dataset = MaskedCBOWDataset(train_windows, word2idx, unk_idx)
train_loader = DataLoader(train_dataset, batch_size=128)

# Example usage of the generator

#for context, target in gen:
#    print(context, target)

#/eg


### create model architecture

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class word2vec(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(28 * 28, 256)  # First hidden layer (784 pixel slots, gradually reducing down)
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 10) # Output layer (64 -> 10, one for each valid prediction)

    def forward(self, x):  # feed forward
        x = x.view(-1, 28 * 28)  # Flatten input from (batch, 1, 28, 28) -> (batch, 784), applies to the tensor prepared above in the dataloader
        x = F.relu(self.fc1(x))  # Activation function (ReLU), no negatives, play with leaky ReLU later
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x
    
loss_function = NN.CrossEntropyLoss()  # using built-in loss function


model = word2vec() ##create the model as described abvoe

optimizer = optim.Adam(model.parameters(), lr=0.001) ### lr = learning rate, 0.001 is apparently a "normal" value. Adam is the optimizer chosen, also fairly default



##### do training

num_epochs = 30 ## passes through the dataset

for epoch in range(num_epochs):
    for images, lables in train_loader: #note uses batches defined earlier
        optimizer.zero_grad() ### reset gradients each time

        outputs = model(images) # forward pass
        loss = loss_function(outputs, lables)

        loss.backward() ## backprop method created by pytorch crossentropyloss function, very convenient
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



### / training


### train model