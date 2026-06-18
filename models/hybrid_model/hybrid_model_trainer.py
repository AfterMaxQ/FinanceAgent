"""
HybridModel 多任务训练脚本
 - 数据：`datas/stock_datas_vif.csv` + FinBERT 768 维文本向量
 - 特征：使用 `selected_features.md` 中保留的 18 个特征顺序作为 LSTM 输入
 - 目标：Target_DA（二分类）与 Target_Magnitude（平滑绝对对数收益）
 - 损失：0.5*BCELoss + 0.5*HuberLoss
 - 评估：方向准确率（DA）与反标后的 RMSE
 - 早停：基于验证方向准确率（Val DA）
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# 允许作为脚本执行时找到项目内模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hybrid_model.hybrid_model import (  # noqa: E402
    HybridModel,
    save_scalers_to_npz,
)


logger = logging.getLogger("hybrid_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# VIF 筛选后的 18 个特征，保持与 selected_features.md 顺序一致
VIF_FEATURES: List[str] = [
    "Volume",
    "GSPC_Close",
    "GSPC_LogReturn",
    "LogReturn",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "Intraday_Range",
    "Trend_Strength",
    "Candle_Body",
    "Month_Sin",
    "Sentiment_Score",
    "BBB_20_2.0_2.0",
    "BBP_20_2.0_2.0",
    "ATR_14",
    "OBV",
    "Beta_60",
    "Alpha_60",
    "Sharpe_60",
]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_features(csv_path: Path) -> pd.DataFrame:
    """
    读取 CSV，生成 Target_DA 与 Target_Magnitude（平滑绝对对数收益）。
    保证时间顺序，避免未来信息泄露。
    """
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Ticker" if "Ticker" in df.columns else "Date"]).reset_index(drop=True)

    if "LogReturn" not in df.columns:
        raise ValueError("缺少 LogReturn 列，无法生成目标。")

    df["Target_DA"] = (df["LogReturn"] > 0).astype(np.float32)

    mag = df["LogReturn"].abs()
    mag_smooth = mag.ewm(span=3, adjust=False, min_periods=1).mean()
    df["Target_Magnitude"] = mag_smooth.astype(np.float32)
    return df


def build_text_embeddings(
    texts: List[str],
    finbert_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """
    使用本地 FinBERT 模型生成 768 维文本向量；对每条 NewsTitles 取 CLS 隐状态。
    """
    tokenizer = AutoTokenizer.from_pretrained(finbert_dir)
    model = AutoModel.from_pretrained(finbert_dir).to(device)
    model.eval()

    all_vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls_vec = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
            all_vecs.append(cls_vec.cpu().numpy())
    return np.vstack(all_vecs).astype(np.float32)


class SequenceHybridDataset(Dataset):
    """基于时间窗口的多任务数据集，返回 (scalar_seq, text_seq, target_da, target_mag_scaled)。"""

    def __init__(
        self,
        scalar_scaled: np.ndarray,
        text_scaled: np.ndarray,
        target_da: np.ndarray,
        target_mag_scaled: np.ndarray,
        seq_len: int,
        start_idx: int,
        end_idx: int,
    ):
        self.scalar_scaled = scalar_scaled
        self.text_scaled = text_scaled
        self.target_da = target_da
        self.target_mag_scaled = target_mag_scaled
        self.seq_len = seq_len
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.n_samples = self.end_idx - self.start_idx
        if self.n_samples <= 0:
            raise ValueError("数据集为空，请检查起止索引或数据量。")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        abs_idx = self.start_idx + idx
        l, r = abs_idx - self.seq_len, abs_idx
        scalar_seq = self.scalar_scaled[l:r]
        text_seq = self.text_scaled[l:r]
        da = self.target_da[abs_idx]
        mag = self.target_mag_scaled[abs_idx]
        return (
            torch.from_numpy(scalar_seq).float(),
            torch.from_numpy(text_seq).float(),
            torch.tensor(da, dtype=torch.float32),
            torch.tensor(mag, dtype=torch.float32),
        )


def build_datasets(
    df: pd.DataFrame,
    text_embeddings: np.ndarray,
    seq_len: int,
    train_ratio: float,
) -> Tuple[SequenceHybridDataset, SequenceHybridDataset, dict]:
    """
    拆分训练/验证（按时间顺序），拟合 scaler（仅用训练部分），并返回数据集与 scaler 字典。
    """
    total_rows = len(df)
    usable = total_rows - seq_len
    if usable <= 0:
        raise ValueError("数据行数不足以构建序列，请增加数据或减小 seq_len。")

    split = int(usable * train_ratio)
    train_end = seq_len + split
    val_start = train_end

    feature_cols = VIF_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    scalar_train = df.iloc[:train_end][feature_cols].values.astype(np.float32)
    text_train = text_embeddings[:train_end].astype(np.float32)
    mag_train = df.iloc[:train_end]["Target_Magnitude"].values.astype(np.float32)

    scalar_scaler = RobustScaler().fit(np.nan_to_num(scalar_train))
    text_scaler = RobustScaler().fit(np.nan_to_num(text_train))
    target_scaler = RobustScaler().fit(mag_train.reshape(-1, 1))

    scalar_all = scalar_scaler.transform(np.nan_to_num(df[feature_cols].values.astype(np.float32)))
    text_all = text_scaler.transform(np.nan_to_num(text_embeddings.astype(np.float32)))
    target_da = df["Target_DA"].values.astype(np.float32)
    target_mag_scaled = target_scaler.transform(df["Target_Magnitude"].values.astype(np.float32).reshape(-1, 1)).reshape(-1)

    train_ds = SequenceHybridDataset(
        scalar_scaled=scalar_all,
        text_scaled=text_all,
        target_da=target_da,
        target_mag_scaled=target_mag_scaled,
        seq_len=seq_len,
        start_idx=seq_len,
        end_idx=train_end,
    )
    val_ds = SequenceHybridDataset(
        scalar_scaled=scalar_all,
        text_scaled=text_all,
        target_da=target_da,
        target_mag_scaled=target_mag_scaled,
        seq_len=seq_len,
        start_idx=val_start,
        end_idx=total_rows,
    )

    scaler_dict = {
        "scalar_scaler": scalar_scaler,
        "text_scaler": text_scaler,
        "target_scaler": target_scaler,
        "sequence_length": seq_len,
        "scalar_feature_cols": feature_cols,
        "text_feature_cols": [f"emb_{i}" for i in range(text_embeddings.shape[1])],
    }
    return train_ds, val_ds, scaler_dict


def compute_metrics(
    probs: np.ndarray,
    targets_da: np.ndarray,
    preds_mag_scaled: np.ndarray,
    targets_mag_scaled: np.ndarray,
    target_scaler: RobustScaler,
) -> Tuple[float, float, float]:
    """计算方向准确率、DA Loss (BCE) 和 RMSE"""
    da_pred = (probs >= 0.5).astype(np.float32)
    da_acc = float((da_pred == targets_da).mean())
    
    # 计算 DA Loss (BCE Loss)
    # 使用数值稳定的方式计算 BCE
    epsilon = 1e-7
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    da_loss = float(-np.mean(targets_da * np.log(probs_clipped) + (1 - targets_da) * np.log(1 - probs_clipped)))

    preds_mag = target_scaler.inverse_transform(preds_mag_scaled.reshape(-1, 1)).reshape(-1)
    true_mag = target_scaler.inverse_transform(targets_mag_scaled.reshape(-1, 1)).reshape(-1)
    rmse = math.sqrt(float(np.mean((preds_mag - true_mag) ** 2)))
    return da_acc, da_loss, rmse


def run_epoch(
    model: HybridModel,
    loader: DataLoader,
    bce_loss: nn.Module,
    huber_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_scaler: RobustScaler,
) -> Tuple[float, float, float, float]:
    """返回: train_loss, da_acc, da_loss, rmse"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_batches = 0.0, 0
    all_probs, all_da, all_mag_pred, all_mag_true = [], [], [], []

    for scalar_seq, text_seq, tgt_da, tgt_mag in loader:
        scalar_seq = scalar_seq.to(device)
        text_seq = text_seq.to(device)
        tgt_da = tgt_da.to(device)
        tgt_mag = tgt_mag.to(device)

        if is_train:
            optimizer.zero_grad()

        prob, mag_pred = model(scalar_seq, text_seq)
        loss_da = bce_loss(prob, tgt_da)
        loss_mag = huber_loss(mag_pred, tgt_mag)
        loss = 0.5 * loss_da + 0.5 * loss_mag

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        all_probs.append(prob.detach().cpu().numpy())
        all_da.append(tgt_da.detach().cpu().numpy())
        all_mag_pred.append(mag_pred.detach().cpu().numpy())
        all_mag_true.append(tgt_mag.detach().cpu().numpy())

    probs_np = np.concatenate(all_probs)
    da_np = np.concatenate(all_da)
    mag_pred_np = np.concatenate(all_mag_pred)
    mag_true_np = np.concatenate(all_mag_true)

    da_acc, da_loss, rmse = compute_metrics(probs_np, da_np, mag_pred_np, mag_true_np, target_scaler)
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, da_acc, da_loss, rmse


def train(
    args: argparse.Namespace,
):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("读取并准备特征...")
    df = prepare_features(Path(args.csv_path))
    logger.info("直接在训练脚本内部生成文本向量（无需外部缓存文件）...")
        texts = df["NewsTitles"].fillna("").astype(str).tolist()
        embeddings = build_text_embeddings(
            texts=texts,
            finbert_dir=Path(args.finbert_dir),
            device=device,
            batch_size=args.text_batch_size,
            max_length=args.text_max_length,
        )

    if embeddings.shape[0] != len(df):
        raise ValueError(f"文本向量行数 {embeddings.shape[0]} 与 CSV 行数 {len(df)} 不一致。")

    train_ds, val_ds, scaler_dict = build_datasets(df, embeddings, args.sequence_length, args.train_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_config = {
        "scalar_input_dim": len(VIF_FEATURES),
        "text_input_dim": embeddings.shape[1],
    }
    model = HybridModel(**model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    huber_loss = nn.HuberLoss(delta=args.huber_delta)

    # 以验证集 Val Loss = 0.7 * DA Loss + 0.3 * RMSE 作为最佳模型与早停标准
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "hybrid_model.pt"
    scaler_path = save_dir / "hybrid_scalers.npz"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_da, train_da_loss, train_rmse = run_epoch(
            model, train_loader, bce_loss, huber_loss, optimizer, device, scaler_dict["target_scaler"]
        )
        _, val_da, val_da_loss, val_rmse = run_epoch(
            model, val_loader, bce_loss, huber_loss, None, device, scaler_dict["target_scaler"]
        )
        
        # 计算验证集 Val Loss = 0.7 * DA Loss + 0.3 * RMSE
        val_loss = 0.7 * val_da_loss + 0.3 * val_rmse

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} (DA {train_da:.4f}, RMSE {train_rmse:.4f}) | "
            f"Val Loss: {val_loss:.4f} (DA {val_da:.4f}, DA Loss {val_da_loss:.4f}, RMSE {val_rmse:.4f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                },
                model_path,
            )
            save_scalers_to_npz(scaler_dict, scaler_path)
            logger.info(f"验证集 Val Loss 降至 {best_val_loss:.4f} (DA Loss {val_da_loss:.4f}, RMSE {val_rmse:.4f})，已保存最佳模型与 scaler 至 {save_dir}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"早停触发（patience={args.patience}），最佳验证 Val Loss {best_val_loss:.4f}")
                break

    logger.info("训练完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HybridModel 多任务训练脚本")
    parser.add_argument("--csv-path", type=str, default="datas/stock_datas_vif.csv", help="输入特征 CSV 路径")
    parser.add_argument("--finbert-dir", type=str, default="models/yiyanghkust_finbert-tone", help="本地 FinBERT 模型目录")
    parser.add_argument("--text-batch-size", type=int, default=32, help="生成文本向量的批大小")
    parser.add_argument("--text-max-length", type=int, default=256, help="生成文本向量的截断长度")
    parser.add_argument("--output-dir", type=str, default="models/hybrid_model/checkpoints", help="模型与 scaler 保存目录")
    parser.add_argument("--sequence-length", type=int, default=30, help="时间序列窗口长度")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例（按时间顺序切分）")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小，要求 128")
    parser.add_argument("--epochs", type=int, default=50, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="HuberLoss delta")
    parser.add_argument("--patience", type=int, default=15, help="早停 patience（验证 Val Loss）")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu，默认自动检测")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

