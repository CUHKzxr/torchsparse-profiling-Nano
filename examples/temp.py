import torch

t = torch.tensor([[1,2,3],[1,2,3]])
print(torch.sum(t,0))
print(torch.sum(t,1))