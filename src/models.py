"""
models.py — Heterogeneous GNN regressor for motor-asymmetry prediction.

Architecture
------------
  1. HBN encoder  : N stacked HeteroRGCNLayer
       - separate message passing for ``intra`` and ``inter`` edges
       - relation-specific parameters (mode='hetero') or shared (mode='homo')
  2. Readout      : per-node-type concatenation of mean and max pooling
  3. Prediction   : two-layer MLP with sigmoid output in (0, 1)
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import (MessagePassing, GraphNorm,
                                global_mean_pool, global_max_pool)
from torch_geometric.utils import add_self_loops


# ── Weighted graph convolution ──────────────────────────────────────────────

class WeightedGCNConv(MessagePassing):
    """Graph convolution with learnable per-edge multiplicative gating.

    Given scalar edge weights ``w_uv``, the message from ``u`` to ``v`` is
    ``(W · x_u) ⊙ σ(U · w_uv)``, where ``σ`` is the logistic function and
    ``U`` a learnable linear projection of the edge weight to the output
    dimension. Messages are aggregated by mean.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_proj = nn.Linear(1, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.lin.weight)

    def reset_parameters(self):
        """Re-initialise every parameter, including ``self.bias``.

        ``self.bias`` is a Parameter attached directly to the wrapper (not
        inside a sub-module), so ``model.apply(reset)`` misses it unless
        this method is defined explicitly.
        """
        nn.init.xavier_uniform_(self.lin.weight)
        self.edge_proj.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor = None) -> Tensor:
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor = None) -> Tensor:
        if edge_attr is not None:
            w = self.edge_proj(edge_attr)
            return x_j * torch.sigmoid(w)
        return x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.bias is not None:
            return aggr_out + self.bias
        return aggr_out


# ── Heterogeneous relational layer ──────────────────────────────────────────

class HeteroRGCNLayer(nn.Module):
    """One layer of message passing on the bilateral heterogeneous graph.

    ``ipsi`` and ``contra`` nodes are updated separately using two
    branches:

      * inter  : messages from the other hemisphere along inter-edges
      * intra  : messages within the same hemisphere along intra-edges

    The two branches are combined as ``α · inter + (1 − α) · intra``,
    optionally passed through a residual, then LayerNorm, PReLU
    activation and dropout.
    """

    def __init__(self, in_ch, out_ch, dropout=0.3, residual=True,
                 mode='hetero', alpha=0.5):
        super().__init__()
        self.residual = residual
        self.mode = mode
        self.alpha = float(alpha)

        if mode == 'hetero':
            self.conv_inter = WeightedGCNConv(in_ch, out_ch)
            self.conv_intra = WeightedGCNConv(in_ch, out_ch)
        else:
            # 'homo' : the two branches share parameters (used as ablation).
            shared = WeightedGCNConv(in_ch, out_ch)
            self.conv_inter = shared
            self.conv_intra = shared

        self.bn_ipsi = nn.LayerNorm(out_ch)
        self.bn_contra = nn.LayerNorm(out_ch)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout)

        if residual:
            self.res_ipsi = nn.Linear(in_ch, out_ch)
            self.res_contra = nn.Linear(in_ch, out_ch)

    def forward(self, x_ipsi, x_contra, data):
        """Return updated ``(x_ipsi, x_contra)`` node features."""
        new = {}
        for ntype, x, x_other, bn, res_lin in [
            ('ipsi', x_ipsi, x_contra, self.bn_ipsi,
             getattr(self, 'res_ipsi', None)),
            ('contra', x_contra, x_ipsi, self.bn_contra,
             getattr(self, 'res_contra', None)),
        ]:
            other = 'contra' if ntype == 'ipsi' else 'ipsi'

            # Inter-hemispheric messages : from ``other`` to ``ntype``.
            etype_back = (other, 'inter', ntype)
            if etype_back in data.edge_types:
                ei_back = data[etype_back].edge_index
                ea_back = data[etype_back].get('edge_attr', None)
                inter_out = self.conv_inter(x_other, ei_back, ea_back)
            else:
                inter_out = torch.zeros(
                    x.size(0), self.conv_inter.lin.weight.size(0),
                    device=x.device)

            # Intra-hemispheric messages : within ``ntype``. Self-loops are
            # added so that nodes with no intra-neighbour still receive
            # their own signal under 'mean' aggregation.
            etype_intra = (ntype, 'intra', ntype)
            if etype_intra in data.edge_types:
                ei_intra = data[etype_intra].edge_index
                ea_intra = data[etype_intra].get('edge_attr', None)
                ei_intra, ea_intra = add_self_loops(
                    ei_intra, ea_intra,
                    fill_value=1.0, num_nodes=x.size(0),
                )
                intra_out = self.conv_intra(x, ei_intra, ea_intra)
            else:
                intra_out = torch.zeros_like(inter_out)

            merged = self.alpha * inter_out + (1.0 - self.alpha) * intra_out
            if self.residual and res_lin is not None:
                merged = merged + res_lin(x)
            # GraphNorm needs the batch tensor to compute per-graph stats;
            # LayerNorm and BatchNorm1d do not — the isinstance switch
            # keeps the layer swappable.
            if isinstance(bn, GraphNorm):
                normed = bn(merged, data[ntype].batch)
            else:
                normed = bn(merged)
            merged = self.act(normed)
            new[ntype] = self.drop(merged)

        return new['ipsi'], new['contra']


# ── Full model : encoder + readout + regression head ────────────────────────

class HeteroGraphRegressor(nn.Module):
    """Heterogeneous GNN regressor for a scalar target in (0, 1).

    Parameters
    ----------
    in_dim : int
        Input node feature dimension (3 for [vol, elongation, curvature]).
    hidden_dims : list[int]
        Hidden dimensions of the encoder stack. Defaults to ``[16, 8]``.
    dropout : float
        Dropout applied after activation in each encoder layer and in the
        MLP head.
    mode : {'hetero', 'homo'}
        'hetero' assigns distinct parameters to the inter and intra
        branches; 'homo' shares them (used as an ablation).
    alpha : float
        Mixing coefficient between the inter and intra messages
        (``h_v = φ(α · m_inter + (1 − α) · m_intra + b)``).
    """

    def __init__(self, in_dim=3, hidden_dims=None, dropout=0.3,
                 mode='hetero', alpha=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        dims = [in_dim] + hidden_dims
        self.encoder = nn.ModuleList([
            HeteroRGCNLayer(dims[i], dims[i + 1],
                            dropout=dropout, mode=mode, alpha=alpha)
            for i in range(len(dims) - 1)
        ])

        hd_last = hidden_dims[-1]
        # MLP head : the readout has dimension 4·hd_last (mean+max per
        # node type), and is projected to a scalar in (0, 1).
        self.head = nn.Sequential(
            nn.Linear(hd_last * 4, hd_last),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hd_last, 1),
            nn.Sigmoid(),
        )

    def _encode(self, data):
        """Run the encoder stack. Return per-node embeddings for both types."""
        xi = data['ipsi'].x
        xc = data['contra'].x
        for layer in self.encoder:
            xi, xc = layer(xi, xc, data)
        return xi, xc

    def _readout(self, xi, xc, data):
        """Concatenate mean and max pooled node embeddings per node type.

        Output dimension is ``4 · hidden_dims[-1]``, in the order
        ``[μ_ipsi, M_ipsi, μ_contra, M_contra]``. The max component
        preserves the most discriminative node signal, while the mean
        captures the distributed pattern; concatenating both prevents
        dilution when a single node carries most of the information.
        """
        batch_i = (data['ipsi'].batch if hasattr(data['ipsi'], 'batch')
                   else torch.zeros(xi.size(0), dtype=torch.long,
                                    device=xi.device))
        batch_c = (data['contra'].batch if hasattr(data['contra'], 'batch')
                   else torch.zeros(xc.size(0), dtype=torch.long,
                                    device=xc.device))
        pi = torch.cat([global_mean_pool(xi, batch_i),
                        global_max_pool(xi, batch_i)], dim=-1)
        pc = torch.cat([global_mean_pool(xc, batch_c),
                        global_max_pool(xc, batch_c)], dim=-1)
        return torch.cat([pi, pc], dim=-1)

    def forward(self, data):
        """Return the predicted score ratio in (0, 1) for each graph."""
        xi, xc = self._encode(data)
        emb = self._readout(xi, xc, data)
        return self.head(emb).squeeze(-1)

    def encode(self, data):
        """Return the graph-level embedding without the regression head."""
        xi, xc = self._encode(data)
        return self._readout(xi, xc, data)
