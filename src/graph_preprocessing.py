"""
graph_preprocessing.py — Dataset, DataLoader and LOO utilities.

Loads pre-built per-patient ``HeteroData`` graphs from a directory of
``*.pkl`` files, exposes a PyTorch Geometric-compatible DataLoader, and
provides a fold-wise standardisation utility that prevents information
leakage from the held-out test patient into the training statistics.
"""
import os
import pickle

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader


class HeteroGraphDataset(Dataset):
    """Load a directory of pre-built ``HeteroData`` graphs.

    Each ``*.pkl`` file is expected to contain a dictionary
    ``{'graph': HeteroData, 'label': float, 'subject': str}``.

    Feature standardisation is not applied at load-time. Call
    :meth:`normalize_for_fold` at the start of every LOO fold to z-score
    node features using training-only statistics.
    """

    def __init__(self, graph_dir):
        super().__init__()
        self.graphs = []
        self.labels = []
        self.subjects = []
        for fname in sorted(f for f in os.listdir(graph_dir)
                            if f.endswith('_hetero.pkl')):
            with open(os.path.join(graph_dir, fname), 'rb') as fh:
                obj = pickle.load(fh)
            subject = obj.get('subject', fname)
            g = _sanitize(obj['graph'], subject=subject)
            self.graphs.append(g)
            self.labels.append(float(obj['label']))
            self.subjects.append(subject)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

        # Keep an untouched copy of the raw node features so that
        # normalize_for_fold() can restart from the raw values on every
        # fold rather than accumulating successive normalisations.
        self._raw_x_ipsi = [g['ipsi'].x.clone() for g in self.graphs]
        self._raw_x_contra = [g['contra'].x.clone() for g in self.graphs]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def normalize_for_fold(self, train_idx):
        """Z-score node features using only the training patients.

        Statistics are computed per ``(node_index, feature_dim)`` and pooled
        across ``ipsi`` and ``contra`` sides. Features are restored to their
        raw values before applying the new normalisation, so successive
        calls with different ``train_idx`` do not accumulate.
        """
        # 1. Restore raw features (cancels any prior normalisation).
        for i, g in enumerate(self.graphs):
            g['ipsi'].x = self._raw_x_ipsi[i].clone()
            g['contra'].x = self._raw_x_contra[i].clone()

        # 2. Statistics computed on the training patients only, pooling
        #    ipsi + contra sides.
        train_xi = torch.stack([self._raw_x_ipsi[i] for i in train_idx])
        train_xc = torch.stack([self._raw_x_contra[i] for i in train_idx])
        x_all = torch.cat([train_xi, train_xc], dim=0)   # (2·|train|, N, F)
        mean = x_all.mean(dim=0)                         # (N, F)
        std = x_all.std(dim=0)                           # (N, F)
        # A channel that is constantly zero by construction (e.g. curvature
        # on subcortical nodes) has std=0; forcing std=1 leaves it at zero
        # after subtraction of the mean.
        std = torch.where(std > 1e-6, std, torch.ones_like(std))

        # 3. Apply the same transform to all graphs (train + test).
        for g in self.graphs:
            g['ipsi'].x = (g['ipsi'].x - mean) / std
            g['contra'].x = (g['contra'].x - mean) / std


def _sanitize(data: HeteroData, subject: str = '') -> HeteroData:
    """Replace NaN and Inf in features and edge weights.

    Node features are replaced by ``0`` and edge weights by ``1`` (neutral
    edge). A short message is printed so that the user is aware that a
    given patient had missing structures.
    """
    warned = False
    for ntype in data.node_types:
        x = data[ntype].x
        if torch.isnan(x).any() or torch.isinf(x).any():
            data[ntype].x = torch.nan_to_num(x, nan=0.0, posinf=0.0,
                                             neginf=0.0)
            warned = True
    for etype in data.edge_types:
        ea = data[etype].get('edge_attr', None)
        if ea is not None and (torch.isnan(ea).any()
                               or torch.isinf(ea).any()):
            data[etype].edge_attr = torch.nan_to_num(ea, nan=1.0,
                                                    posinf=1.0, neginf=1.0)
            warned = True
    if warned:
        print(f"  [sanitize] {subject} : NaN/Inf replaced "
              f"(missing structure?)")
    return data


def get_loader(dataset, indices, batch_size=4, shuffle=True):
    """Return a PyG DataLoader over the given subset of indices."""
    subset = Subset(dataset, indices)
    return PyGDataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def loo_splits(n):
    """Leave-one-out splitter. Yields ``(train_indices, [test_index])``."""
    all_idx = list(range(n))
    for i in all_idx:
        yield [j for j in all_idx if j != i], [i]
