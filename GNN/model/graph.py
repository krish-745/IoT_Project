import numpy as np
import torch


def build_topk_graph(embeddings, top_k=5, exclude_self=True):
    n = embeddings.shape[0]
    normed = torch.nn.functional.normalize(embeddings, dim=1)
    sim    = normed @ normed.T
    if exclude_self:
        sim.fill_diagonal_(-1e9)
    top_k = min(top_k, n - 1)
    _, idx = sim.topk(top_k, dim=1)
    src = torch.arange(n).unsqueeze(1).expand(-1, top_k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)
