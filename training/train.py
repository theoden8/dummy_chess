#!/usr/bin/env python3
"""
Efficient NNUE Training using PyTorch

Architecture: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1
"""

import argparse
import struct
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm

# ============================================================================
# Architecture Constants (must match NNUE.hpp)
# ============================================================================

HALFKP_SIZE = 41024
FT_OUT = 256
L1_OUT = 32
L2_OUT = 32

FT_QUANT_SCALE = 127
WEIGHT_QUANT_SCALE = 64

PIECE_TO_INDEX = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [-1, -1],  # P, N, B, R, Q, K
]


def get_halfkp_features(fen: str):
    """Extract HalfKP features from FEN."""
    board = chess.Board(fen)
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)

    white_feats, black_feats = [], []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type == chess.KING:
            continue
        pt = pc.piece_type - 1
        is_white = pc.color

        white_feats.append(
            wk * 641 + PIECE_TO_INDEX[pt][0 if is_white else 1] * 64 + sq + 1
        )
        black_feats.append(
            (63 - bk) * 641
            + PIECE_TO_INDEX[pt][1 if is_white else 0] * 64
            + (63 - sq)
            + 1
        )

    return white_feats, black_feats, 0 if board.turn else 1


# ============================================================================
# Model
# ============================================================================


class NNUE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(HALFKP_SIZE, FT_OUT, mode="sum", sparse=True)
        self.ft_bias = torch.nn.Parameter(torch.zeros(FT_OUT))
        self.l1 = torch.nn.Linear(FT_OUT * 2, L1_OUT)
        self.l2 = torch.nn.Linear(L1_OUT, L2_OUT)
        self.out = torch.nn.Linear(L2_OUT, 1)

        torch.nn.init.normal_(self.ft.weight, std=0.01)
        for m in [self.l1, self.l2, self.out]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)

    def forward(self, w_idx, w_off, b_idx, b_off, stm):
        w_ft = torch.clamp(self.ft(w_idx, w_off) + self.ft_bias, 0, 1)
        b_ft = torch.clamp(self.ft(b_idx, b_off) + self.ft_bias, 0, 1)
        ft = torch.where(
            stm.unsqueeze(1) == 0,
            torch.cat([w_ft, b_ft], 1),
            torch.cat([b_ft, w_ft], 1),
        )
        x = torch.clamp(self.l1(ft), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        return self.out(x)


# ============================================================================
# Dataset
# ============================================================================


class ChunkDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, chunk_size: int = 50000):
        self.path = path
        self.chunk_size = chunk_size
        self._len: int | None = None

    def __len__(self) -> int:
        """Count rows in file (cached)."""
        if self._len is None:
            import polars as pl

            self._len = pl.scan_csv(self.path).select(pl.len()).collect().item()
        return self._len

    def __iter__(self):
        compression = "gzip" if self.path.endswith(".gz") else None
        for chunk in pd.read_csv(
            self.path, chunksize=self.chunk_size, compression=compression
        ):
            for _, row in chunk.iterrows():
                try:
                    w, b, stm = get_halfkp_features(row["fen"])
                    yield w, b, stm, float(row["score"])
                except:
                    continue


def collate_sparse(batch):
    w_all, b_all, w_off, b_off = [], [], [0], [0]
    stm_list, score_list = [], []
    for w, b, stm, score in batch:
        w_all.extend(w)
        b_all.extend(b)
        w_off.append(len(w_all))
        b_off.append(len(b_all))
        stm_list.append(stm)
        score_list.append(score)
    return (
        torch.tensor(w_all, dtype=torch.long),
        torch.tensor(w_off[:-1], dtype=torch.long),
        torch.tensor(b_all, dtype=torch.long),
        torch.tensor(b_off[:-1], dtype=torch.long),
        torch.tensor(stm_list, dtype=torch.long),
        torch.tensor(score_list, dtype=torch.float32).unsqueeze(1),
    )


# ============================================================================
# Training
# ============================================================================


class Tracker:
    """
    Training metrics tracker with epoch averaging.

    Tracks metrics by split (e.g., 'train', 'val') and name. Uses defaultdict(list)
    for consistent tracking logic across all metrics.

    Usage:
        tracker = Tracker()
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                loss = train_step(batch)
                tracker.track("train", "loss", loss.item())
                pbar.set_postfix(**tracker.postfix)
            # Run validation
            for batch in val_loader:
                tracker.track("val", "loss", compute_val_loss(batch))
            tracker.submit_epoch()
    """

    def __init__(self):
        from collections import defaultdict

        self._current: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._history: list[dict[str, dict[str, float]]] = []
        self._epoch = 0

    def track(self, split: str, name: str, value: float) -> None:
        """Track a metric value for the current epoch."""
        self._current[split][name].append(value)

    @property
    def postfix(self) -> dict[str, str]:
        """Get current aggregate values for tqdm postfix."""
        result = {}
        for split, metrics in self._current.items():
            for name, values in metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    result[f"{split}_{name}"] = f"{avg:.4f}"
        return result

    def submit_epoch(self) -> dict[str, dict[str, float]]:
        """
        Finalize current epoch: average all tracked metrics, print summary.

        Returns:
            Dict of {split: {name: avg_value}} for all tracked metrics
        """
        self._epoch += 1

        # Average all tracked metrics
        metrics: dict[str, dict[str, float]] = {"epoch": {"n": float(self._epoch)}}
        for split, split_metrics in self._current.items():
            metrics[split] = {}
            for name, values in split_metrics.items():
                metrics[split][name] = sum(values) / len(values) if values else 0.0

        # Store in history and reset
        self._history.append(metrics)
        self._current.clear()

        # Print epoch summary
        parts = [f"Epoch {self._epoch}"]
        for split, split_metrics in metrics.items():
            if split == "epoch":
                continue
            for name, value in split_metrics.items():
                parts.append(f"{split}_{name}: {value:.4f}")
        print(" - ".join(parts))

        return metrics

    def __getitem__(self, key: str) -> dict[str, float]:
        """Get last epoch's metrics for a split (e.g., tracker['train'])."""
        if self._history:
            return self._history[-1].get(key, {})
        return {}

    @property
    def history(self) -> list[dict[str, dict[str, float]]]:
        """Get full training history."""
        return self._history

    @property
    def epoch(self) -> int:
        """Get current epoch number."""
        return self._epoch

    def best(self, split: str, name: str) -> float:
        """Get best (minimum) value for a metric across all epochs."""
        values = [
            m[split][name]
            for m in self._history
            if split in m and name in m[split] and not np.isnan(m[split][name])
        ]
        return min(values) if values else float("inf")


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            w_idx, w_off, b_idx, b_off, stm, target = [x.to(device) for x in batch]
            pred = model(w_idx, w_off, b_idx, b_off, stm)
            total_loss += F.mse_loss(pred, target).item()
            n += 1
    return total_loss / max(n, 1)


def train(
    train_dataset,
    val_dataset,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[NNUE, Tracker]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sparse,
        num_workers=4,
        prefetch_factor=2,
    )
    n_train_batches = (len(train_dataset) + batch_size - 1) // batch_size
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_sparse,
            num_workers=2,
        )
        if val_dataset is not None
        else None
    )

    model = NNUE().to(device)
    tracker = Tracker()

    # Use SparseAdam for sparse embedding, AdamW for dense layers
    sparse_params = [model.ft.weight]
    dense_params = [p for n, p in model.named_parameters() if "ft.weight" not in n]

    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr)
    dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, weight_decay=1e-4)

    # Use CosineAnnealingLR which steps per epoch (no fixed steps_per_epoch needed)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dense_optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()

        pbar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=n_train_batches
        )
        for batch in pbar:
            w_idx, w_off, b_idx, b_off, stm, target = [x.to(device) for x in batch]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = F.mse_loss(model(w_idx, w_off, b_idx, b_off, stm), target)
                scaler.scale(loss).backward()
                scaler.step(sparse_optimizer)
                scaler.step(dense_optimizer)
                scaler.update()
            else:
                loss = F.mse_loss(model(w_idx, w_off, b_idx, b_off, stm), target)
                loss.backward()
                sparse_optimizer.step()
                dense_optimizer.step()

            tracker.track("train", "loss", loss.item())
            pbar.set_postfix(**tracker.postfix)

        # Run validation
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    w_idx, w_off, b_idx, b_off, stm, target = [
                        x.to(device) for x in batch
                    ]
                    pred = model(w_idx, w_off, b_idx, b_off, stm)
                    val_loss = F.mse_loss(pred, target).item()
                    tracker.track("val", "loss", val_loss)

        scheduler.step()
        tracker.submit_epoch()

        val_loss = tracker["val"].get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output.replace(".nnue", ".pt"))
            export_nnue(model, output)
            print(f"  -> saved (best val: {val_loss:.4f})")

    return model, tracker


# ============================================================================
# Export
# ============================================================================


def export_nnue(model: NNUE, path: str):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x7AF32F20))
        f.write(struct.pack("<I", 0))
        arch = b"Features=HalfKP(Friend)[41024->256x2]->[32->32]->1"
        f.write(struct.pack("<I", len(arch)))
        f.write(arch)
        f.write(struct.pack("<I", 0x5D69D5B9))

        # Feature transformer
        bias = (
            (model.ft_bias.detach().cpu().numpy() * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(bias.tobytes())
        weight = (
            (model.ft.weight.detach().cpu().numpy().T * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(weight.T.tobytes())

        f.write(struct.pack("<I", 0))

        # Hidden layers
        for layer, scale in [
            (model.l1, FT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
            (model.l2, WEIGHT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
        ]:
            f.write(
                (layer.bias.detach().cpu().numpy() * scale).astype(np.int32).tobytes()
            )
            f.write(
                (layer.weight.detach().cpu().numpy().T * WEIGHT_QUANT_SCALE)
                .clip(-128, 127)
                .astype(np.int8)
                .tobytes()
            )

        # Output
        f.write(
            (model.out.bias.detach().cpu().numpy() * WEIGHT_QUANT_SCALE**2)
            .astype(np.int32)
            .tobytes()
        )
        f.write(
            (model.out.weight.detach().cpu().numpy().flatten() * WEIGHT_QUANT_SCALE)
            .clip(-128, 127)
            .astype(np.int8)
            .tobytes()
        )

    print(f"Exported: {path}")


# ============================================================================
# Evaluation
# ============================================================================


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model's parameters."""
    return next(model.parameters()).device


def evaluate_fen(
    fen: str, model: NNUE | None = None, model_path: str | None = None
) -> float:
    """
    Evaluate a FEN position using the NNUE model.

    Args:
        fen: FEN string of the position to evaluate
        model: Pre-loaded NNUE model (optional)
        model_path: Path to .pt model file (used if model is None)

    Returns:
        Evaluation score in centipawns from the side to move's perspective
    """
    if model is None:
        if model_path is None:
            model_path = str(Path(__file__).parent / "network.pt")
        model = NNUE()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

    device = _get_model_device(model)
    w_feats, b_feats, stm = get_halfkp_features(fen)

    with torch.no_grad():
        w_idx = torch.tensor(w_feats, dtype=torch.long, device=device)
        w_off = torch.tensor([0], dtype=torch.long, device=device)
        b_idx = torch.tensor(b_feats, dtype=torch.long, device=device)
        b_off = torch.tensor([0], dtype=torch.long, device=device)
        stm_t = torch.tensor([stm], dtype=torch.long, device=device)

        score = model(w_idx, w_off, b_idx, b_off, stm_t).item()

    return score


def evaluate_fens(
    fens: list[str], model: NNUE | None = None, model_path: str | None = None
) -> list[float]:
    """
    Evaluate multiple FEN positions using the NNUE model (batched).

    Args:
        fens: List of FEN strings to evaluate
        model: Pre-loaded NNUE model (optional)
        model_path: Path to .pt model file (used if model is None)

    Returns:
        List of evaluation scores in centipawns from each side to move's perspective
    """
    if model is None:
        if model_path is None:
            model_path = str(Path(__file__).parent / "network.pt")
        model = NNUE()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

    device = _get_model_device(model)

    # Extract features for all positions
    batch = [get_halfkp_features(fen) for fen in fens]

    # Collate into batched tensors
    w_all, b_all, w_off, b_off = [], [], [0], [0]
    stm_list = []
    for w, b, stm in batch:
        w_all.extend(w)
        b_all.extend(b)
        w_off.append(len(w_all))
        b_off.append(len(b_all))
        stm_list.append(stm)

    with torch.no_grad():
        w_idx = torch.tensor(w_all, dtype=torch.long, device=device)
        w_off_t = torch.tensor(w_off[:-1], dtype=torch.long, device=device)
        b_idx = torch.tensor(b_all, dtype=torch.long, device=device)
        b_off_t = torch.tensor(b_off[:-1], dtype=torch.long, device=device)
        stm_t = torch.tensor(stm_list, dtype=torch.long, device=device)

        scores = model(w_idx, w_off_t, b_idx, b_off_t, stm_t).squeeze(-1).tolist()

    return scores


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv.gz")
    parser.add_argument("--val", default="data/val.csv.gz")
    parser.add_argument("--output", default="network.nnue")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_dataset = ChunkDataset(args.train)
    val_dataset = ChunkDataset(args.val) if Path(args.val).exists() else None
    train(
        train_dataset, val_dataset, args.output, args.epochs, args.batch_size, args.lr
    )
