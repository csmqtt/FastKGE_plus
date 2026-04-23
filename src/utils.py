import random
import os, sys
from copy import deepcopy
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F
import numpy as np
from prettytable import PrettyTable
import loralib

""" Set random seeds """
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

""" Training precision helper.

The original paper stores all embeddings in FP64 (`.double()`). On consumer
GPUs this is a massive hidden bottleneck: RTX 30xx/40xx FP64 throughput is
~1/32 of FP32, and the KGE forward is matmul-dominated. Routing every
`.double()` site through this helper lets us flip the whole codebase to FP32
with `-precision fp32` (default) while still being able to reproduce the
paper exactly with `-precision fp64`.
"""
def model_dtype(args) -> torch.dtype:
    p = str(getattr(args, "precision", "fp32")).lower()
    return torch.float64 if p in ("fp64", "double", "float64") else torch.float32

""" Get learnable parameters """
def get_param(shape, args=None):
    dtype = model_dtype(args) if args is not None else torch.float64
    param = Parameter(torch.Tensor(*shape).to(dtype=dtype))
    xavier_normal_(param.data)
    return param

""" Calculate infoNCE loss """
def infoNCE(embeds1, embeds2, nodes, temp=0.1):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return (-torch.log(nume / deno)).mean()