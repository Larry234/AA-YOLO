import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x = torch.Tensor(np.arange(25).reshape((5, -1)))
t = 1
x = x[None, None, ...]
print(x)
x = x.unfold(2, 2, 2).unfold(3, 2, 2)
print(x)
B, C, o_h, o_w = x.size()[:4]
w = F.softmax(x.contiguous().view(B, C, o_h, o_w, -1) / t, dim=-1)
w = w.view(B, C, o_h, o_w, 2, 2)
print(w)
print(F.softmax(torch.Tensor([12, 13, 17, 18]), dim=0))