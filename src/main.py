"""
main.py — Training entry-point for the heterogeneous GNN motor-asymmetry model.

Runs leave-one-out (LOO) cross-validation over multiple random seeds on
the given cohort of pre-built heterogeneous graphs, and saves predictions
and metrics to an Excel workbook.

Two operating modes are provided:
  1. Pre-configured : `--score {bbt,nhpt}` selects the corresponding
     graph directory (data/bbt or data/nhpt) and the tuned hyper-parameters
     for that target.
  2. Manual : pass `--graph_dir` and all training hyper-parameters
     explicitly.

Example:
    python src/main.py --score bbt --out_xlsx predictions_bbt.xlsx
    python src/main.py --score nhpt --out_xlsx predictions_nhpt.xlsx
"""
import argparse
import datetime
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from graph_preprocessing import HeteroGraphDataset, get_loader, loo_splits
from models import HeteroGraphRegressor
from metrics_utils import compute_extended_metrics, metrics_to_dataframe


# ── Pre-configured hyper-parameters per prediction target ───────────────────
# Values obtained by multi-objective Optuna search (Spearman ρ maximised,
# MAE minimised) over 10 LOO seeds per trial. See the accompanying paper
# for details on the search space and selection procedure.
HPS_BY_SCORE = {
    'bbt': dict(
        graph_dir='data/bbt',
        dropout=0.0730,
        lr=0.00060,
        weight_decay=2.63e-06,
        batch_size=4,
        epochs=100,
    ),
    'nhpt': dict(
        graph_dir='data/nhpt',
        dropout=0.1415,
        lr=0.00141,
        weight_decay=2.86e-07,
        batch_size=16,
        epochs=120,
    ),
}

# Ten seeds used for LOO repetitions (kept fixed so that runs are
# reproducible across users of the released code).
SEEDS = [1, 7, 10, 31, 42, 100, 123, 256, 777, 999]


# ── Small helpers ───────────────────────────────────────────────────────────

def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU + CUDA) from a single value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_lr(optimizer, factor):
    """Multiply every parameter group's learning rate by `factor`."""
    for pg in optimizer.param_groups:
        pg['lr'] *= factor


# ── Training loop for a single LOO fold ─────────────────────────────────────

def train_fold(model, dataset, train_idx, test_idx, args, device):
    """Train the model on a single LOO fold and predict the held-out patient.

    Features are z-scored per node index using training-fold statistics
    only (no leakage into the held-out test patient). The learning rate
    is halved at 40 % and 80 % of the total number of epochs.
    """
    model.to(device)
    dataset.normalize_for_fold(train_idx)

    train_loader = get_loader(dataset, train_idx,
                              batch_size=args.batch_size, shuffle=True)
    test_loader = get_loader(dataset, test_idx,
                             batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    e1 = int(args.epochs * 0.4)
    e2 = int(args.epochs * 0.8)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            pred = model(batch)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in (e1, e2):
            adjust_lr(optimizer, 0.5)

    model.eval()
    pred_val, true_val = None, None
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_val = float(pred.item())
            true_val = float(labels.item())
    return pred_val, true_val


# ── Full LOO × seeds run and Excel export ───────────────────────────────────

def run(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    dataset = HeteroGraphDataset(args.graph_dir)
    n = len(dataset)
    print(f"Dataset : {n} patients")

    all_rho, all_p, all_mae, all_r2 = [], [], [], []
    rows = []  # one row per (seed, fold) — full audit trail for the xlsx

    for seed in SEEDS:
        set_seed(seed)
        preds, trues = [], []
        print(f"\n── seed={seed} ──────────────────────────────────────────",
              flush=True)
        for fold_idx, (train_idx, test_idx) in enumerate(loo_splits(n), 1):
            # A fresh model instance per fold guarantees strict independence
            # between folds (no parameter leakage from the previous fold).
            model = HeteroGraphRegressor(
                in_dim=args.in_dim,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                mode=args.mode,
                alpha=args.alpha,
            )
            subj = dataset.subjects[test_idx[0]]
            p, t = train_fold(model, dataset, train_idx, test_idx, args, device)
            preds.append(p)
            trues.append(t)
            ae = abs(p - t) if (p == p and t == t) else float('nan')
            rows.append({'seed': seed, 'fold': fold_idx, 'subject': subj,
                         'pred': p, 'true': t, 'abs_err': ae})
            print(f"  fold {fold_idx:2d}/{n}  subj={subj:<10s}  "
                  f"pred={p:.3f}  true={t:.3f}  |err|={ae:.3f}", flush=True)

        pa, ta = np.array(preds), np.array(trues)
        rho, pval = spearmanr(ta, pa)
        mae = float(np.mean(np.abs(pa - ta)))
        r2 = float(r2_score(ta, pa))
        all_rho.append(rho)
        all_p.append(pval)
        all_mae.append(mae)
        all_r2.append(r2)
        print(f"  → seed={seed:4d}  ρ={rho:.3f}  p={pval:.4f}  "
              f"MAE={mae:.4f}  R²={r2:.4f}", flush=True)

    print(f"\n── Per-seed averages ({len(SEEDS)} seeds) ─────────────────────")
    print(f"  ρ   = {np.mean(all_rho):.3f} ± {np.std(all_rho):.3f}")
    print(f"  MAE = {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"  R²  = {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")

    # ── Excel export : four sheets ──────────────────────────────────────
    #   predictions      : raw predictions (one row per seed × fold)
    #   metrics          : per-seed ρ, p-value, MAE, R² + mean/std
    #   mean_predictions : predictions averaged across seeds per subject +
    #                      extended metrics (MAE with bootstrap CI, CCC,
    #                      tolerance thresholds, etc.)
    #   config           : hyper-parameters and metadata for reproducibility
    df_pred = pd.DataFrame(rows)
    df_metrics = pd.DataFrame({
        'seed': list(SEEDS) + ['mean', 'std'],
        'rho':  all_rho + [float(np.mean(all_rho)), float(np.std(all_rho))],
        'p':    all_p   + [float(np.mean(all_p)),   float(np.std(all_p))],
        'mae':  all_mae + [float(np.mean(all_mae)), float(np.std(all_mae))],
        'r2':   all_r2  + [float(np.mean(all_r2)),  float(np.std(all_r2))],
    })

    mean_per_subj = df_pred.groupby('subject', as_index=False).agg(
        true=('true', 'first'),
        pred_mean=('pred', 'mean'),
    )
    mean_per_subj['abs_err'] = (mean_per_subj['pred_mean']
                                - mean_per_subj['true']).abs()
    ta_m = mean_per_subj['true'].to_numpy()
    pa_m = mean_per_subj['pred_mean'].to_numpy()
    metrics_mp = compute_extended_metrics(ta_m, pa_m)
    df_mean_metrics = metrics_to_dataframe(metrics_mp)

    print(f"\n── Metrics on seed-averaged predictions ─────────────────────")
    print(f"  MAE          = {metrics_mp['mae']:.4f}  "
          f"[95% CI {metrics_mp['mae_ci_low']:.4f}"
          f"–{metrics_mp['mae_ci_high']:.4f}]")
    print(f"  medAE        = {metrics_mp['medae']:.4f}")
    print(f"  RMSE         = {metrics_mp['rmse']:.4f}")
    print(f"  bias         = {metrics_mp['bias']:+.4f}")
    print(f"  |err| ≤ 0.02 : {metrics_mp['pct_within_0.02']:5.1f}%")
    print(f"  |err| ≤ 0.03 : {metrics_mp['pct_within_0.03']:5.1f}%")
    print(f"  CCC (Lin)    = {metrics_mp['ccc']:.3f}")
    print(f"  ρ Spearman   = {metrics_mp['rho_spearman']:.3f}  "
          f"(p={metrics_mp['p_spearman']:.4f})")

    config_rows = [{'parameter': k, 'value': str(v)}
                   for k, v in vars(args).items()]
    config_rows.append({'parameter': 'seeds', 'value': str(list(SEEDS))})
    config_rows.append({'parameter': 'n_patients', 'value': str(n)})
    config_rows.append({'parameter': 'timestamp',
                        'value': datetime.datetime.now()
                                          .isoformat(timespec='seconds')})
    df_config = pd.DataFrame(config_rows)

    with pd.ExcelWriter(args.out_xlsx) as writer:
        df_pred.to_excel(writer, sheet_name='predictions', index=False)
        df_metrics.to_excel(writer, sheet_name='metrics', index=False)
        mean_per_subj.to_excel(writer, sheet_name='mean_predictions',
                               index=False, startrow=0)
        df_mean_metrics.to_excel(writer, sheet_name='mean_predictions',
                                 index=False, startrow=len(mean_per_subj) + 3)
        df_config.to_excel(writer, sheet_name='config', index=False)
    print(f"\n  → predictions saved to : {args.out_xlsx}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train the heterogeneous GNN motor-asymmetry model with "
            "LOO cross-validation and save predictions to an Excel file."))
    p.add_argument('--score', choices=['bbt', 'nhpt'], default=None,
                   help="Prediction target. When set, automatically selects "
                        "the corresponding graph directory (data/bbt or "
                        "data/nhpt) and the tuned hyper-parameters. Any "
                        "hyper-parameter explicitly passed on the command "
                        "line overrides this default.")
    p.add_argument('--graph_dir', default=None,
                   help="Directory containing the per-patient .pkl "
                        "HeteroData graphs. Ignored when --score is set "
                        "unless explicitly provided.")
    p.add_argument('--in_dim', type=int, default=3,
                   help="Input feature dimension per node "
                        "([vol, elongation, curvature]).")
    p.add_argument('--hidden_dims', type=int, nargs='+', default=[16, 8],
                   help="Hidden dimensions of the encoder and MLP head.")
    p.add_argument('--mode', default='hetero', choices=['hetero', 'homo'],
                   help="'hetero' uses distinct parameters per relation "
                        "(intra / inter). 'homo' uses shared parameters.")
    p.add_argument('--alpha', type=float, default=0.5,
                   help="Mixing coefficient between the two relational "
                        "branches: h_v = phi(alpha * m_inter + "
                        "(1 - alpha) * m_intra + b).")
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--device', default='cpu',
                   choices=['cpu', 'cuda', 'auto'],
                   help="'cpu' is the default because the graphs are tiny "
                        "(18 nodes) and GPU overhead outweighs its benefit.")
    p.add_argument('--out_xlsx', default='predictions.xlsx',
                   help="Output Excel workbook.")
    return p


def resolve_args(args):
    """Fill in defaults from HPS_BY_SCORE when --score is set, unless the
    user already passed a value on the command line. Any hyper-parameter
    still unset after this falls back to the NHPT defaults."""
    if args.score is not None:
        defaults = HPS_BY_SCORE[args.score]
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    fallback = HPS_BY_SCORE['nhpt']
    for key in ('graph_dir', 'dropout', 'lr', 'weight_decay',
                'batch_size', 'epochs'):
        if getattr(args, key, None) is None:
            setattr(args, key, fallback[key])
    return args


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_args(args)
    print(args)
    run(args)
