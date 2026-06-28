"""多会话存储层 (Conversation Store)
============================================

负责把「聊天历史」从单一的 `st.session_state[chat_key]` 升级为
**多会话 + 磁盘持久化** 的资产，使前端可以：

- 列出已有的对话
- 加载/切换某一历史对话
- 新建对话、重命名、删除

设计目标
---------
1. **纯数据层**：不依赖 Streamlit，不依赖 LLM，便于单测与脚本复用。
2. **轻量持久化**：默认以 JSON 文件落盘到 `.cache/conversations.json`，
   失败时静默降级为「仅内存」模式，保证前端不被 IO 异常拖垮。
3. **稳定 API**：以 `conversation_id` 为主键，全部增删改查围绕它进行。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Conversation:
    """一次完整的人机对话。"""

    id: str
    title: str
    ticker: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ----- 序列化 ----- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Conversation":
        return cls(
            id=str(raw.get("id") or uuid.uuid4().hex),
            title=str(raw.get("title") or "未命名对话"),
            ticker=str(raw.get("ticker") or ""),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
            messages=list(raw.get("messages") or []),
            meta=dict(raw.get("meta") or {}),
        )

    # ----- 便捷操作 ----- #
    def touch(self) -> None:
        self.updated_at = _now_iso()

    def append_message(self, role: str, content: str, **extra: Any) -> None:
        msg: Dict[str, Any] = {"role": role, "content": content}
        if extra:
            msg.update(extra)
        self.messages.append(msg)
        self.touch()

    def replace_messages(self, messages: Iterable[Dict[str, Any]]) -> None:
        self.messages = list(messages)
        self.touch()

    def message_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") in ("user", "assistant"))

    def preview(self, max_len: int = 32) -> str:
        """取首条用户消息作为预览。"""
        for m in self.messages:
            if m.get("role") == "user":
                text = str(m.get("content") or "").strip().replace("\n", " ")
                if len(text) > max_len:
                    text = text[: max_len - 1] + "…"
                return text or "（空对话）"
        return "（尚未提问）"


# --------------------------------------------------------------------------- #
# 存储管理器
# --------------------------------------------------------------------------- #
class ConversationStore:
    """多会话存储。线程安全（使用 RLock 保护内存视图与文件写入）。"""

    DEFAULT_PATH = Path(".cache") / "conversations.json"

    def __init__(self, storage_path: Optional[Path | str] = None):
        self._path: Optional[Path] = Path(storage_path) if storage_path else self.DEFAULT_PATH
        self._lock = threading.RLock()
        self._conversations: Dict[str, Conversation] = {}
        self._load_from_disk()

    # ---------- 持久化 ---------- #
    def _load_from_disk(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("conversations") if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                return
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                conv = Conversation.from_dict(entry)
                self._conversations[conv.id] = conv
            logger.info("已从 %s 加载 %d 个历史对话", self._path, len(self._conversations))
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载会话存储失败（将以空状态启动）: %s", exc)

    def _flush_to_disk(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "conversations": [c.to_dict() for c in self._conversations.values()],
            }
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入会话存储失败（仅内存生效）: %s", exc)

    # ---------- 查询 ---------- #
    def list(self) -> List[Conversation]:
        """按更新时间倒序返回所有对话。"""
        with self._lock:
            return sorted(
                self._conversations.values(),
                key=lambda c: c.updated_at,
                reverse=True,
            )

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._conversations.get(conversation_id)

    def __contains__(self, conversation_id: str) -> bool:
        return conversation_id in self._conversations

    def __len__(self) -> int:
        return len(self._conversations)

    # ---------- 增删改 ---------- #
    def create(
        self,
        ticker: str,
        title: Optional[str] = None,
        initial_messages: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        now = _now_iso()
        conv = Conversation(
            id=uuid.uuid4().hex,
            title=title or _default_title(ticker),
            ticker=ticker or "",
            created_at=now,
            updated_at=now,
            messages=list(initial_messages or []),
            meta=dict(meta or {}),
        )
        with self._lock:
            self._conversations[conv.id] = conv
            self._flush_to_disk()
        return conv

    def save(self, conversation: Conversation) -> None:
        """覆盖式更新整个 Conversation（messages 等都以传入对象为准）。"""
        with self._lock:
            conversation.touch()
            self._conversations[conversation.id] = conversation
            self._flush_to_disk()

    def update_messages(
        self,
        conversation_id: str,
        messages: Iterable[Dict[str, Any]],
    ) -> Optional[Conversation]:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.replace_messages(messages)
            self._flush_to_disk()
            return conv

    def rename(self, conversation_id: str, new_title: str) -> Optional[Conversation]:
        new_title = (new_title or "").strip() or "未命名对话"
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.title = new_title
            conv.touch()
            self._flush_to_disk()
            return conv

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            existed = self._conversations.pop(conversation_id, None) is not None
            if existed:
                self._flush_to_disk()
            return existed

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()
            self._flush_to_disk()


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_title(ticker: str) -> str:
    base = (ticker or "新对话").strip().upper() or "新对话"
    ts = datetime.now().strftime("%m-%d %H:%M")
    return f"{base} · {ts}"
