import torch
import torch.nn as nn

emb = nn.Embedding(10, 5)
print(emb.weight.shape)
