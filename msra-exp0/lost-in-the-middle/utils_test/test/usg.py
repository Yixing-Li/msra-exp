import torch

a = torch.Tensor([[1,2], [3,4], [5, 6]])
b = torch.Tensor([[1,2], [3,4], [5, 6]])
print(a.mul(b))
print(torch.mul(a, b))