from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import DataLoader, Dataset


GROUPS = [range(1, 6), range(6, 11), range(11, 16), range(16, 21)]
PICKUP_TO_DROPOFF = {0: 2, 1: 3}
LEGAL_GROUP_ORDERS = [
    p for p in itertools.permutations(range(4))
    if p.index(0) < p.index(2) and p.index(1) < p.index(3)
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_historical_data(repo: Path, cache_root: Path) -> np.ndarray:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / "chengdu_data.npy"
    if not cache.exists():
        ref = "f5ef14cd3335053bf6e3ad3b4fb451a9e7972103"
        spec = f"{ref}:sim_data/chengdu_data.npy"
        with cache.open("wb") as f:
            cp = subprocess.run(
                ["git", "-C", str(repo), "show", spec],
                stdout=f,
                stderr=subprocess.PIPE,
                check=False,
            )
        if cp.returncode != 0:
            cache.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to recover historical chengdu_data.npy: "
                + cp.stderr.decode("utf-8", errors="replace")
            )
    return np.load(cache, allow_pickle=True)


def road_distance_matrix(repo: Path, cache_root: Path) -> np.ndarray:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / "chengdu_directed_shortest_distance_matrix.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    links = pd.read_csv(repo / "sim_data/chengdu_link-mod.txt")
    n = max(int(links.Node_Start.max()), int(links.Node_End.max())) + 1
    graph = csr_matrix(
        (
            links.Length.to_numpy(float),
            (links.Node_Start.to_numpy(int), links.Node_End.to_numpy(int)),
        ),
        shape=(n, n),
    )
    dist = dijkstra(graph, directed=True, return_predecessors=False)
    np.save(cache, dist)
    return dist


def route_length(dist: np.ndarray, point_ids: np.ndarray, solution: np.ndarray) -> float:
    ids = point_ids[solution.astype(int)]
    return float(sum(dist[ids[i], ids[i + 1]] for i in range(len(ids) - 1)))


def group_of_index(idx: int) -> int | None:
    if 1 <= idx <= 5:
        return 0
    if 6 <= idx <= 10:
        return 1
    if 11 <= idx <= 15:
        return 2
    if 16 <= idx <= 20:
        return 3
    return None


def legal_mask(selected_groups: list[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(21, dtype=torch.bool, device=device)
    done = set(selected_groups)
    if not selected_groups:
        allowed_groups = [0, 1]
    else:
        allowed_groups = []
        for g in range(4):
            if g in done:
                continue
            if g == 2 and 0 not in done:
                continue
            if g == 3 and 1 not in done:
                continue
            allowed_groups.append(g)
    for g in allowed_groups:
        mask[list(GROUPS[g])] = True
    return mask


def compute_angle_features(points: np.ndarray) -> np.ndarray:
    # Historical tensors store latitude/longitude in the first two columns.
    xy = points[:, :2].astype(np.float64)
    feats = []
    for group in GROUPS:
        center = xy[list(group)].mean(axis=0)
        delta = center[None, :] - xy
        # Angle with due north. Use atan2(east, north), then sin/cos to avoid wraparound.
        theta = np.arctan2(delta[:, 1], delta[:, 0])
        feats.append(np.sin(theta)[:, None])
        feats.append(np.cos(theta)[:, None])
    return np.concatenate(feats, axis=1).astype(np.float32)


def exact_optimum(dist: np.ndarray, point_ids: np.ndarray) -> float:
    ids = point_ids.astype(int)
    best = math.inf
    for order in LEGAL_GROUP_ORDERS:
        a, b, c, d = [np.asarray(list(GROUPS[g]), dtype=int) for g in order]
        c0 = dist[ids[0], ids[a]][:, None, None, None]
        c1 = dist[ids[a][:, None], ids[b][None, :]][:, :, None, None]
        c2 = dist[ids[b][:, None], ids[c][None, :]][None, :, :, None]
        c3 = dist[ids[c][:, None], ids[d][None, :]][None, None, :, :]
        best = min(best, float(np.min(c0 + c1 + c2 + c3)))
    return best


def certify_targets(data: np.ndarray, dist: np.ndarray) -> dict:
    exact = 0
    target_route_error = []
    optimality_gap = []
    for sample in data:
        ids = np.asarray(sample["Points_id"], dtype=int)
        target = np.asarray(sample["Solutions"], dtype=int)
        target_length = float(sample["Opt_Length"])
        recomputed = route_length(dist, ids, target)
        optimum = exact_optimum(dist, ids)
        target_route_error.append(abs(recomputed - target_length))
        optimality_gap.append((target_length - optimum) / max(optimum, 1e-12) * 100.0)
        if abs(target_length - optimum) <= max(1e-4, 1e-7 * max(1.0, optimum)):
            exact += 1
    return {
        "n": len(data),
        "target_route_mean_abs_error": float(np.mean(target_route_error)),
        "target_route_max_abs_error": float(np.max(target_route_error)),
        "target_exact_optimum_count": int(exact),
        "target_exact_optimum_fraction": float(exact / len(data)),
        "target_vs_exact_gap_mean_pct": float(np.mean(optimality_gap)),
        "target_vs_exact_gap_max_pct": float(np.max(optimality_gap)),
    }


class RouteDataset(Dataset):
    def __init__(self, data: np.ndarray, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, use_angle: bool):
        self.data = data
        self.indices = np.asarray(indices, dtype=int)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.use_angle = use_angle

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        sample = self.data[int(self.indices[k])]
        x = np.asarray(sample["Points"], dtype=np.float32)
        x = (x - self.mean) / self.std
        if self.use_angle:
            raw = np.asarray(sample["Points"], dtype=np.float32)
            x = np.concatenate([x, compute_angle_features(raw)], axis=1)
        return {
            "x": torch.from_numpy(x),
            "solution": torch.as_tensor(sample["Solutions"], dtype=torch.long),
            "point_ids": torch.as_tensor(sample["Points_id"], dtype=torch.long),
            "opt_length": torch.as_tensor(float(sample["Opt_Length"]), dtype=torch.float32),
        }


class ConstrainedPointer(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 160, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.position = nn.Embedding(21, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.init_state = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        pos = torch.arange(n, device=x.device)[None, :].expand(b, -1)
        h = self.input_proj(x) + self.position(pos)
        return self.encoder(h)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        teacher_forcing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encode(x)
        pooled = enc.mean(dim=1)
        state = self.init_state(torch.cat([enc[:, 0], pooled], dim=-1))
        prev = enc[:, 0]
        batch = x.size(0)
        selected_groups: list[list[int]] = [[] for _ in range(batch)]
        logits_steps = []
        picks = []

        for step in range(4):
            state = self.gru(prev, state)
            raw = self.score(torch.tanh(self.key(enc) + self.query(state)[:, None, :])).squeeze(-1)
            legal = torch.stack(
                [legal_mask(selected_groups[b], x.device) for b in range(batch)],
                dim=0,
            )
            masked = raw.masked_fill(~legal, -1e9)
            logits_steps.append(masked)
            if teacher_forcing and targets is not None:
                chosen = targets[:, step + 1]
            else:
                chosen = masked.argmax(dim=1)
            picks.append(chosen)
            for b, idx in enumerate(chosen.detach().cpu().tolist()):
                g = group_of_index(int(idx))
                if g is None:
                    raise RuntimeError(f"Illegal decoder choice {idx}")
                selected_groups[b].append(g)
            prev = enc[torch.arange(batch, device=x.device), chosen]

        logits = torch.stack(logits_steps, dim=1)
        pred = torch.stack(picks, dim=1)
        return logits, pred


@dataclass
class Metrics:
    mean_gap_pct: float
    median_gap_pct: float
    p95_gap_pct: float
    max_gap_pct: float
    exact_sequence_acc_pct: float
    selected_point_acc_pct: float
    invalid_route_pct: float


def is_legal(pred: np.ndarray) -> bool:
    if len(pred) != 5 or int(pred[0]) != 0:
        return False
    groups = [group_of_index(int(v)) for v in pred[1:]]
    if any(g is None for g in groups) or sorted(groups) != [0, 1, 2, 3]:
        return False
    assert all(g is not None for g in groups)
    return groups.index(0) < groups.index(2) and groups.index(1) < groups.index(3)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dist: np.ndarray,
    device: torch.device,
) -> Metrics:
    model.eval()
    gaps = []
    exact = 0
    point_ok = 0
    invalid = 0
    total = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        _, p = model(x, teacher_forcing=False)
        p = p.cpu().numpy()
        ids = batch["point_ids"].numpy()
        true = batch["solution"].numpy()
        opt = batch["opt_length"].numpy()
        for i in range(len(p)):
            pred = np.concatenate([[0], p[i].astype(int)])
            plen = route_length(dist, ids[i], pred)
            gaps.append((plen - float(opt[i])) / max(float(opt[i]), 1e-12) * 100.0)
            exact += int(np.array_equal(pred, true[i]))
            point_ok += sum(ids[i][pred[j]] == ids[i][true[i][j]] for j in range(1, 5))
            invalid += int(not is_legal(pred))
            total += 1
    return Metrics(
        mean_gap_pct=float(np.mean(gaps)),
        median_gap_pct=float(np.median(gaps)),
        p95_gap_pct=float(np.percentile(gaps, 95)),
        max_gap_pct=float(np.max(gaps)),
        exact_sequence_acc_pct=100.0 * exact / total,
        selected_point_acc_pct=100.0 * point_ok / (4 * total),
        invalid_route_pct=100.0 * invalid / total,
    )


def nearest_legal_baseline(data: np.ndarray, indices: np.ndarray, dist: np.ndarray) -> Metrics:
    gaps = []
    exact = 0
    point_ok = 0
    invalid = 0
    for idx in indices:
        s = data[int(idx)]
        ids = np.asarray(s["Points_id"], dtype=int)
        true = np.asarray(s["Solutions"], dtype=int)
        opt = float(s["Opt_Length"])
        selected_groups: list[int] = []
        pred = [0]
        for _ in range(4):
            mask = legal_mask(selected_groups, torch.device("cpu")).numpy()
            candidates = np.flatnonzero(mask)
            cur_node = ids[pred[-1]]
            costs = dist[cur_node, ids[candidates]]
            chosen = int(candidates[int(np.argmin(costs))])
            pred.append(chosen)
            selected_groups.append(int(group_of_index(chosen)))
        pred_arr = np.asarray(pred, dtype=int)
        plen = route_length(dist, ids, pred_arr)
        gaps.append((plen - opt) / max(opt, 1e-12) * 100.0)
        exact += int(np.array_equal(pred_arr, true))
        point_ok += sum(ids[pred_arr[j]] == ids[true[j]] for j in range(1, 5))
        invalid += int(not is_legal(pred_arr))
    n = len(indices)
    return Metrics(
        mean_gap_pct=float(np.mean(gaps)),
        median_gap_pct=float(np.median(gaps)),
        p95_gap_pct=float(np.percentile(gaps, 95)),
        max_gap_pct=float(np.max(gaps)),
        exact_sequence_acc_pct=100.0 * exact / n,
        selected_point_acc_pct=100.0 * point_ok / (4 * n),
        invalid_route_pct=100.0 * invalid / n,
    )


def train_one(
    data: np.ndarray,
    dist: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    variant: str,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[dict, dict]:
    set_seed(seed)
    train_points = np.concatenate(
        [np.asarray(data[int(i)]["Points"], dtype=np.float32) for i in train_idx],
        axis=0,
    )
    mean = train_points.mean(axis=0)
    std = train_points.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    use_angle = variant == "angle"
    input_dim = 23 + (8 if use_angle else 0)

    train_ds = RouteDataset(data, train_idx, mean, std, use_angle)
    val_ds = RouteDataset(data, val_idx, mean, std, use_angle)
    test_ds = RouteDataset(data, test_idx, mean, std, use_angle)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConstrainedPointer(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best = None
    best_val = math.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["solution"].to(device, non_blocking=True)
            logits, _ = model(x, targets=target, teacher_forcing=True)
            loss = sum(F.cross_entropy(logits[:, t, :], target[:, t + 1]) for t in range(4)) / 4.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, dist, device)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": asdict(val_metrics),
        }
        history.append(rec)
        print(
            f"seed={seed} variant={variant} epoch={epoch} "
            f"loss={rec['train_loss']:.5f} val_gap={val_metrics.mean_gap_pct:.4f} "
            f"val_exact={val_metrics.exact_sequence_acc_pct:.2f}"
        )

        if val_metrics.mean_gap_pct < best_val - 1e-6:
            best_val = val_metrics.mean_gap_pct
            best_epoch = epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    assert best is not None
    model.load_state_dict(best)
    val_metrics = evaluate(model, val_loader, dist, device)
    test_metrics = evaluate(model, test_loader, dist, device)
    result = {
        "seed": seed,
        "variant": variant,
        "best_epoch": best_epoch,
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
    }
    return result, {"history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", choices=["base", "angle"], default=["base", "angle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get("LITEGD_CACHE", "/workspace/.server-control/litegd-repro-cache"))
    data = load_historical_data(repo, cache_root)
    dist = road_distance_matrix(repo, cache_root)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0))

    cert = certify_targets(data, dist)
    print("CERTIFICATION", json.dumps(cert, sort_keys=True))

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(data))
    train_idx = perm[:800]
    val_idx = perm[800:900]
    test_idx = perm[900:1000]
    np.savez(args.output / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    baseline = nearest_legal_baseline(data, test_idx, dist)
    print("NEAREST_LEGAL_TEST", json.dumps(asdict(baseline), sort_keys=True))

    results = []
    histories = {}
    for variant in args.variants:
        for seed in args.seeds:
            result, extra = train_one(
                data=data,
                dist=dist,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                variant=variant,
                seed=seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            results.append(result)
            histories[f"{variant}-{seed}"] = extra
            print("RESULT", json.dumps(result, sort_keys=True))

    summary = {
        "protocol": {
            "dataset": "historical chengdu_data.npy recovered from Git history",
            "split": "fixed 800/100/100 train/validation/test",
            "split_seed": args.split_seed,
            "train_from_scratch": True,
            "strict_rule_mask": True,
            "selection_metric": "validation mean route gap",
        },
        "certification": cert,
        "nearest_legal_test": asdict(baseline),
        "runs": results,
        "aggregate": {},
    }
    for variant in args.variants:
        subset = [r for r in results if r["variant"] == variant]
        metrics = {}
        for key in asdict(Metrics(0, 0, 0, 0, 0, 0, 0)).keys():
            vals = [r["test"][key] for r in subset]
            metrics[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        summary["aggregate"][variant] = metrics

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print("FINAL_SUMMARY", json.dumps(summary["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
