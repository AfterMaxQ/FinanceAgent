"""
Iteratively compute VIF and drop the highest-VIF feature until all are below
the given threshold (default 5) using datas/stock_datas_cleaned.csv.

Outputs a markdown report to selected_features.md with retained features and
the VIF trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "datas" / "stock_datas_cleaned.csv"
DEFAULT_OUTPUT = ROOT / "selected_features.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIF-based feature selection.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input cleaned CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="VIF cutoff to keep features (use 10 for a looser rule).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional random sample size to speed up VIF computation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def load_numeric_features(path: Path, max_rows: int | None, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows is not None and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=random_state)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # drop constant columns to avoid infinite VIF
    constant_cols = numeric_df.columns[numeric_df.std(ddof=0, numeric_only=True) == 0]
    if len(constant_cols) > 0:
        numeric_df = numeric_df.drop(columns=list(constant_cols))

    # remove rows with missing numeric values for stable VIF computation
    numeric_df = numeric_df.dropna(axis=0)

    # standardize to reduce conditioning issues
    numeric_df = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
    numeric_df = numeric_df.dropna(axis=1)  # drop any col that became NaN after std
    return numeric_df


def compute_vif(df: pd.DataFrame) -> pd.Series:
    arr = df.values
    vifs = []
    for i in range(arr.shape[1]):
        vifs.append(float(variance_inflation_factor(arr, i)))
    return pd.Series(vifs, index=df.columns)


def iterative_vif(df: pd.DataFrame, threshold: float) -> Tuple[List[str], list[dict]]:
    features = list(df.columns)
    history: list[dict] = []
    iteration = 1

    while len(features) > 1:
        vif_series = compute_vif(df[features])
        max_feature = vif_series.idxmax()
        max_value = vif_series.loc[max_feature]

        history.append(
            {
                "iteration": iteration,
                "max_feature": max_feature,
                "max_vif": float(max_value),
                "vif_table": vif_series.sort_values(ascending=False),
            }
        )

        if not np.isfinite(max_value) or max_value > threshold:
            features = [f for f in features if f != max_feature]
            iteration += 1
            continue
        break

    return features, history


def write_report(
    output: Path,
    threshold: float,
    used_rows: int,
    history: list[dict],
    final_features: list[str],
    final_vif_table: pd.Series,
) -> None:
    lines: list[str] = []
    lines.append("# VIF特征筛选报告")
    lines.append("")
    lines.append(f"- 阈值: {threshold}")
    lines.append(f"- 用于计算的样本行数: {used_rows}")
    lines.append(f"- 初始迭代次数: {len(history)}")
    lines.append("")
    lines.append("## 保留的特征")
    for feat in final_features:
        lines.append(f"- {feat}")

    lines.append("")
    lines.append("## 迭代VIF明细（按最大VIF排序）")
    for record in history:
        lines.append(f"### 迭代 {record['iteration']}")
        lines.append(f"- 最大VIF特征: {record['max_feature']}")
        lines.append(f"- 最大VIF值: {record['max_vif']:.4f}")
        lines.append("")
        lines.append("| 特征 | VIF |")
        lines.append("| --- | ---: |")
        for feat, vif in record["vif_table"].items():
            lines.append(f"| {feat} | {vif:.4f} |")
        lines.append("")

    lines.append("## 最终特征VIF")
    lines.append("| 特征 | VIF |")
    lines.append("| --- | ---: |")
    for feat, vif in final_vif_table.items():
        lines.append(f"| {feat} | {vif:.4f} |")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    numeric_df = load_numeric_features(args.input, args.max_rows, args.random_state)
    used_rows = len(numeric_df)

    if numeric_df.empty:
        raise ValueError("No numeric features available after preprocessing.")

    final_features, history = iterative_vif(numeric_df, threshold=args.threshold)

    # recompute VIF on final set for the report
    if len(final_features) == 1:
        final_vif_table = pd.Series([np.nan], index=final_features)
    else:
        final_vif_table = compute_vif(numeric_df[final_features]).sort_values(ascending=False)
    write_report(
        output=args.output,
        threshold=args.threshold,
        used_rows=used_rows,
        history=[
            {
                **rec,
                "vif_table": rec["vif_table"],
            }
            for rec in history
        ],
        final_features=final_features,
        final_vif_table=final_vif_table,
    )

    print(f"Report written to {args.output} with {len(final_features)} kept features.")


if __name__ == "__main__":
    main()

