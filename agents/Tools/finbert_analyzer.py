# agents/Tools/finbert_analyzer.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Optional
import warnings
from pathlib import Path

# 修正：直接在这里定义配置，避免跨模块导入问题
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE_DIR = str(_PROJECT_ROOT / "models") # 关键修复：转换为字符串
MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_LENGTH = 256
LOCAL_MODEL_DIR = _PROJECT_ROOT / "models" / "yiyanghkust_finbert-tone"
SNAPSHOT_ROOT = Path(MODEL_CACHE_DIR) / "models--yiyanghkust--finbert-tone" / "snapshots"


def _find_local_model_path() -> Optional[Path]:
    """寻找包含 vocab 与模型权重的本地路径。"""
    candidates = []
    if LOCAL_MODEL_DIR.exists():
        candidates.append(LOCAL_MODEL_DIR)
    if SNAPSHOT_ROOT.exists():
        candidates.extend([p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir()])
    for path in candidates:
        vocab_file = path / "vocab.txt"
        model_file = path / "model.safetensors"
        if not model_file.exists():
            model_file = path / "pytorch_model.bin"
        if vocab_file.exists() and model_file.exists():
            return path
    return None

class FinBERTAnalyzer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FinBERTAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO [FinBERTAnalyzer]: Initializing on device: {self.device}")

        try:
            local_model_path = _find_local_model_path()
            load_plan = []
            if local_model_path:
                load_plan.append((str(local_model_path), True, f"选择本地快照 {local_model_path}"))
            load_plan.append((MODEL_NAME, False, f"使用远程模型 {MODEL_NAME}"))

            last_error = None
            for source, local_only, info in load_plan:
                try:
                    print(f"INFO [FinBERTAnalyzer]: {info}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        source,
                        cache_dir=MODEL_CACHE_DIR,
                        local_files_only=local_only
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        source,
                        cache_dir=MODEL_CACHE_DIR,
                        local_files_only=local_only
                    )
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"INFO [FinBERTAnalyzer]: 模型 {source} 加载成功。")
                    self._initialized = True
                    break
                except Exception as load_err:
                    last_error = load_err
                    print(f"WARN [FinBERTAnalyzer]: 尝试加载 {source} 失败: {load_err}")

            if not self._initialized:
                raise RuntimeError(f"无法加载 FinBERT 模型: {last_error}") from last_error

        except Exception as e:
            print(f"ERROR [FinBERTAnalyzer]: 模型加载失败: {e}")
            raise RuntimeError(f"无法加载 FinBERT 模型: {e}") from e

    def classify_texts(self, texts: Union[str, List[str]]) -> List[Dict[str, Union[str, float]]]:
        if isinstance(texts, str): texts = [texts]
        if not texts or not all(isinstance(t, str) for t in texts): return []
        
        results = []
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            id2label = self.model.config.id2label
            
            for i in range(len(texts)):
                scores_tensor = probabilities[i]
                prediction_id = torch.argmax(scores_tensor).item()
                label = id2label[prediction_id]
                scores_dict = {id2label[j]: scores_tensor[j].item() for j in range(len(id2label))}
                results.append({"label": label, "scores": scores_dict})
        
        except Exception as e:
            warnings.warn(f"ERROR [FinBERTAnalyzer]: 文本分类时发生错误: {e}")
            return [{"error": str(e)}] * len(texts)
        return results

    def calc_score(self, texts: Union[str, List[str]]) -> float:
        if isinstance(texts, str): texts = [texts]
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts): return 0.0
        
        classifications = self.classify_texts(texts)
        score_map = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0}
        total_score = 0.0
        valid_count = 0
        for result in classifications:
            if "label" in result:
                total_score += score_map.get(result["label"], 0.0)
                valid_count += 1
        return total_score / valid_count if valid_count > 0 else 0.0