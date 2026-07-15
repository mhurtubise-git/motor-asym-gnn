# motor-asym-gnn

Heterogeneous graph neural network for predicting hand motor asymmetry
after neonatal arterial ischemic stroke (NAIS) from bilateral morphometric
descriptors of the motor system.

Each patient is represented as a bilateral heterogeneous graph of 18 nodes
(nine anatomical regions per hemisphere) with intra- and inter-hemispheric
edges. The model predicts a scalar side-imbalance ratio in ``(0, 1)`` for
either the Box-and-Blocks Test (BBT, gross dexterity) or the Nine-Hole Peg
Test (NHPT, fine dexterity), using leave-one-out cross-validation.

## Repository layout

```
motor-asym-gnn/
├── src/
│   ├── main.py                 # training entry-point
│   ├── models.py               # HeteroGraphRegressor + layers
│   ├── graph_preprocessing.py  # dataset, dataloader, LOO utilities
│   └── metrics_utils.py        # extended metrics (MAE, CCC, bootstrap CI)
├── data/
│   ├── bbt/                    # per-patient graphs, BBT ratio label
│   └── nhpt/                   # per-patient graphs, NHPT ratio label
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

## Usage

### Pre-configured runs

The tuned hyper-parameters and matching graph directory for each target
are wired into the ``--score`` option:

```bash
python src/main.py --score bbt  --out_xlsx predictions_bbt.xlsx
python src/main.py --score nhpt --out_xlsx predictions_nhpt.xlsx
```

### Manual runs

Pass any hyper-parameter explicitly to override the defaults:

```bash
python src/main.py \
    --graph_dir data/bbt \
    --dropout 0.10 \
    --lr 1e-3 \
    --epochs 150 \
    --out_xlsx predictions_custom.xlsx
```

### Output

Each run writes an Excel workbook containing four sheets:

  * ``predictions``      : one row per (seed, fold), raw predictions
  * ``metrics``          : per-seed ρ, p-value, MAE, R² + mean and std
  * ``mean_predictions`` : predictions averaged across seeds per subject,
                            with extended metrics (bootstrap CI on MAE and
                            medAE, CCC, ICC, tolerance percentages,
                            Bland-Altman-style bias)
  * ``config``           : hyper-parameters and metadata (seeds, timestamp)

## Model at a glance

* **Nodes** (9 per hemisphere): primary motor cortex (M1), somatosensory
  cortex (S1), premotor + supplementary motor area (PM+SMA), thalamus,
  caudate, putamen, pallidum, cerebral peduncle, cerebellum.
* **Node features** (dimension 3): volume, elongation index, mean
  cortical curvature (zero-padded where not applicable).
* **Intra-hemispheric edges** (12 per hemisphere): the
  cortico-striatal-thalamo-cortical motor loop and its cerebellar
  connection.
* **Inter-hemispheric edges** (4 pairs): homologous M1↔M1 and Cer↔Cer
  connections plus two cross-cerebellar projections to the contralateral
  motor thalamus.
* **Encoder**: one heterogeneous relational layer with distinct parameters
  for the intra and inter branches, fused as
  ``h_v = φ(α · m_inter + (1 − α) · m_intra + b)`` with α = 0.5.
* **Readout**: per-node-type concatenation of mean and max pooling
  (dimension 4·d).
* **Head**: two-layer MLP with sigmoid output.

## Reproducibility

Every LOO run is repeated over the fixed list of ten seeds
``[1, 7, 10, 31, 42, 100, 123, 256, 777, 999]``. Node features are
z-scored per fold using training statistics only (no test-fold leakage).

## Citation

If you use this code, please cite the accompanying paper.
