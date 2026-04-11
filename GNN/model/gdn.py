import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import build_topk_graph


def _scatter_softmax(src, index, num_nodes):
    max_vals = torch.full((num_nodes,), float('-inf'), device=src.device)
    max_vals.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
    src_stable = src - max_vals[index]
    exp_src = torch.exp(src_stable)
    sum_exp = torch.zeros(num_nodes, device=src.device)
    sum_exp.scatter_add_(0, index, exp_src)
    return exp_src / (sum_exp[index] + 1e-8)


class GDNLayer(nn.Module):
    def __init__(self, in_features, out_features, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v   = nn.Linear(embed_dim, 1, bias=False)
        self.fc  = nn.Linear(2 * in_features, out_features)

    def forward(self, x, embeddings, edge_index):
        B, N, W = x.shape
        src, dst = edge_index[0], edge_index[1]
        e_src = self.W_q(embeddings[src])
        e_dst = self.W_k(embeddings[dst])
        score = self.v(torch.tanh(e_src + e_dst)).squeeze(-1)
        attn  = _scatter_softmax(score, dst, num_nodes=N)
        x_src = x[:, src, :]
        weighted = x_src * attn.unsqueeze(0).unsqueeze(-1)
        agg = torch.zeros(B, N, W, device=x.device)
        agg.index_add_(1, dst, weighted)
        out = self.fc(torch.cat([x, agg], dim=-1))
        return F.relu(out)


class GDN(nn.Module):
    def __init__(self, n_sensors, window_size, embed_dim=64, hidden_dim=64, top_k=5, dynamic_graph=True):
        super().__init__()
        self.n_sensors     = n_sensors
        self.window_size   = window_size
        self.top_k         = top_k
        self.dynamic_graph = dynamic_graph
        self.embeddings = nn.Embedding(n_sensors, embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.gdn_layer = GDNLayer(window_size, hidden_dim, embed_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._edge_index = None

    def _get_edge_index(self, device):
        if self.dynamic_graph or self._edge_index is None:
            with torch.no_grad():
                edge_index = build_topk_graph(self.embeddings.weight.detach(), top_k=self.top_k)
            self._edge_index = edge_index.to(device)
        return self._edge_index

    def forward(self, x):
        B, N, W = x.shape
        device  = x.device
        node_ids   = torch.arange(N, device=device)
        embeddings = self.embeddings(node_ids)
        edge_index = self._get_edge_index(device)
        h    = self.gdn_layer(x, embeddings, edge_index)
        pred = self.fc_out(h).squeeze(-1)
        return pred
