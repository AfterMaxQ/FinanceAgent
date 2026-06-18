# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinanceAgent is a multi-modal financial quantitative mining agent that combines DeepSeek LLM reasoning with local PyTorch/Transformers quant models. The project uses a **"Dual Brain"** architecture:

- **Left Brain (LLM)**: `FinancialAgent` (`agents/decision_agent.py`) — DeepSeek V3 via OpenAI SDK for function calling, intent routing, and structured report generation.
- **Right Brain (Quant Tools)**: 7 local analysis tools registered in `agents/Tools/tool_registry.py` — each tool performs specialized financial computation.

The system is exposed through a **Streamlit** frontend (`frontend/app.py`).

## Project Structure Warning

There is a **nested duplicate** `FinanceAgentV2.0/` subdirectory that mirrors the root. This appears to be an accidental nesting artifact. The canonical working directory is the root (`F:\Python\FinanceAgentV2.0`). Always work at the root level, not inside the nested subdirectory.

## Key Commands

### Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Data Pipeline (sequential, each step writes to `datas/stock_datas.csv`)

```bash
# Step 1: Download raw data from Yahoo Finance
python get_data.py --start 2009-01-01 --end 2019-12-31

# Step 2: Merge news data
python merge_news_into_stock_data.py

# Step 3: Feature engineering (technical indicators + FinBERT sentiment)
python feature_engineering.py

# Step 4: Clean data (fill NaN, clip outliers)
python clean_stock_data.py

# Step 5 (optional): VIF feature selection → stock_datas_vif.csv
python vif_feature_selection.py --threshold 5.0
python clean_stock_data_vif.py
```

### Running the App

```bash
streamlit run app.py
# or equivalently:
streamlit run frontend/app.py
```

### Training the Hybrid Model

```bash
python models/hybrid_model/hybrid_model_trainer.py \
  --csv-path datas/stock_datas_vif.csv \
  --finbert-dir models/yiyanghkust_finbert-tone \
  --output-dir models/hybrid_model/checkpoints
```

### Generate requirements.txt from imports

```bash
python generate_requirements.py
```

### Proxy Configuration

`get_data.py` defaults to proxy `http://127.0.0.1:7897`. Comment out the proxy lines in the script if you don't need it.

## Architecture

### Agent Decision Loop (`agents/decision_agent.py`)

`FinancialAgent.run()` builds a system prompt enforcing a strict SOP framework, then enters a function-calling loop (max 10 rounds):

1. **Macro First** — must call `analyze_market_regime` before any stock analysis
2. **Query Qualification** — classifies user intent as short-term trading, long-term investing, or comprehensive
3. **Path Selection** — routes to technical tools (predict, calculate_levels, search_similar_periods) or fundamental tools (analyze_financial_statements)
4. **Synthesis** — cross-validates findings across tools
5. **Report** — outputs in a structured 4-section format (Core View → Macro → Quant → Fundamental → Risk)

The agent uses DeepSeek's API via the OpenAI SDK (`openai.OpenAI` with custom `base_url`). If no API key is provided, the frontend falls back to a rule-engine degraded mode (charts and quant tools work, but no AI conversation).

### Tool Registry (`agents/Tools/tool_registry.py`)

Centralized registry containing:
- `tools`: List of function definitions in DeepSeek API format (name, description, JSON Schema parameters)
- `TOOL_MAPPING`: Dict mapping function names to their Python classes

All 7 tools:

| Function Name | Class | What It Does |
|---|---|---|
| `calc_score` / `classify_texts` | `FinBERTAnalyzer` | FinBERT sentiment (singleton, lazy-loads model) |
| `predict` | `HybridPredictor` | T+1 price: direction probability + magnitude |
| `calculate_levels` | `SupportResistanceScanner` | KDE-based support/resistance levels |
| `search_similar_periods` | `SimilaritySearcher` | Historical K-line pattern matching (Z-score + sliding window) |
| `analyze_market_regime` | `MacroAnalyzer` | ^GSPC/^VIX/^TNX systemic risk assessment |
| `analyze_financial_statements` | `FinancialStatementAnalyzer` | SEC EDGAR 10-K analysis with computed ratios |

Tools are instantiated per-call in `_execute_tool()`, which routes by function name, filters arguments to allowed parameters, and returns JSON strings.

### Hybrid Model (`models/hybrid_model/hybrid_model.py`)

A dual-head deep learning model for T+1 price prediction:

- **Temporal Tower**: 2-layer LSTM processing 18 VIF-selected scalar features (sequence_length=30)
- **Text Tower**: 4-layer Transformer encoder processing 768-dim FinBERT text embeddings
- **Fusion Layer**: Cross-attention with learned gating between temporal and text representations
- **Dual Heads**: Classification head (Sigmoid → P(up)) + Regression head (Huber → magnitude)
- Training loss: `0.5 * BCELoss + 0.5 * HuberLoss`
- Uses `RobustScaler` for all features and targets
- Checkpoints saved as `hybrid_model.pt` + `hybrid_scalers.npz`

### Data Flow

```
stock_statistics.csv → get_data.py → stock_datas.csv (raw OHLCV)
                                         │
analyst_ratings_processed.csv → merge_news_into_stock_data.py
                                         │
                               stock_datas.csv (+ NewsTitles)
                                         │
                             feature_engineering.py (+ 27 features)
                                         │
                               clean_stock_data.py (clip + fill)
                                         │
                               clean_stock_data_vif.py (keep 18 VIF features)
```

VIF-retained features (18): Volume, GSPC_Close, GSPC_LogReturn, LogReturn, MACDh, MACDs, Intraday_Range, Trend_Strength, Candle_Body, Month_Sin, Sentiment_Score, BBB, BBP, ATR_14, OBV, Beta_60, Alpha_60, Sharpe_60.

### FinBERT Integration

`FinBERTAnalyzer` is a **singleton** that lazy-loads `yiyanghkust/finbert-tone`. It searches for the model in `models/yiyanghkust_finbert-tone/` first, then falls back to HuggingFace Hub. Used both by `feature_engineering.py` (batch sentiment scoring during pipeline) and at runtime for on-demand news classification.

## Key Dependencies

- `torch==2.7.0`, `transformers==4.51.3` — deep learning core
- `yfinance==0.2.66` — Yahoo Finance data source
- `streamlit`, `plotly` — frontend
- `pandas-ta` — technical indicators
- `statsmodels` — VIF computation
- `openai` — DeepSeek API client (not in requirements.txt but required for AI features)

## Python Path Dependencies

Many modules add the project root to `sys.path` at import time (e.g., `sys.path.insert(0, str(_project_root))`). This is intentional — the project does not use pip-installable package structure. When adding new modules under `agents/Tools/`, always add the path insertion boilerplate if you need cross-package imports.

## .gitignore Behavior

The `.gitignore` excludes entire `/datas/`, `/models/`, `*.csv`, `*.pkl` — these are not tracked in git. Model downloads and data processing must be done after clone.
