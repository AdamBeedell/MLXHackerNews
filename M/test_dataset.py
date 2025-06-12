from torch.utils.data import DataLoader
from training_data.training_data import SkipGramDataset

# example data
pairs = [(1, 2), (3, 4), (5, 6)]

# Create dataset
dataset = SkipGramDataset(pairs)

# Use a DataLoader to load batches
loader = DataLoader(dataset, batch_size=3, shuffle=False)

# Loop through a few batches
for batch in loader:
    print(batch)