import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.Tensor(np.arange(25).reshape((5, -1)))
x = x[None, None, ...]
x = x.to(device)
x = x.unfold(2, 2, 2).unfold(3, 2, 2)
b, c, oh, ow = x.size()[:4]
x = x.contiguous().view(b, c, oh, ow, -1)
x1 = x[:, :, :, :, 1]
print(x1.size())


