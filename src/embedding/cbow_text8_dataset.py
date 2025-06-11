import torch
from torch.utils.data import IterableDataset, get_worker_info

class CBOWText8Dataset(IterableDataset):
    def __init__(self, file_path, w2i, context_size=2, device='cpu'):
        self.file_path = file_path
        self.w2i       = w2i
        self.unk       = w2i['<unk>']
        self.ctx       = context_size
        self.device    = device

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id   = 0
            num_workers = 1
        else:
            worker_id   = worker_info.id
            num_workers = worker_info.num_workers

        window_idx = 0

        # open the huge file once per worker
        with open(self.file_path, 'r') as f:
            for chunk in f:
                words = chunk.strip().split()
                length = len(words)
                # for every word from word ctx length to the end - the context length
                for i in range(self.ctx, length - self.ctx):
                    if window_idx % num_workers != worker_id:
                        window_idx += 1
                        continue

                    ctx_idxs = [
                        self.w2i.get(words[i+j], self.unk)
                        for j in range(-self.ctx, self.ctx+1) if j != 0
                    ]
                    tgt_idx  = self.w2i.get(words[i], self.unk)

                    yield torch.tensor(ctx_idxs, dtype=torch.long), tgt_idx
                    window_idx += 1
