import torch

a = torch.Tensor([[0.1,0.2],[3,7]])
a_norm = torch.abs(torch.sum(a, dim = -1, keepdim = True))
a_norm = torch.maximum(a_norm, torch.ones_like(a_norm))
b = a / a_norm

b = 2 * a
print(b)
