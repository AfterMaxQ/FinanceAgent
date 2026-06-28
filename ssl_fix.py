"""修复 Windows 证书库损坏导致的 SSL 报错。

现象：
    ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:....)
    出现在 streamlit / tornado 启动，或 requests / yfinance / openai 发起 HTTPS 时。

原因：
    CPython 在 Windows 上的 ssl.SSLContext.load_default_certs() 会把系统证书库
    （CA / ROOT）里的所有证书拼成一大段，一次性交给 load_verify_locations()。
    只要证书库里存在哪怕一张损坏/截断的证书，整批解析就会失败并抛出 ASN1 错误，
    导致所有需要默认证书的 HTTPS 行为崩溃。

修复策略（按优先级）：
    1) 优先加载 certifi 自带的权威 CA 包（requests/yfinance 一般已附带 certifi）；
    2) 否则逐张加载系统证书，跳过那张损坏的证书。

使用方法：
    必须在导入 streamlit / tornado / requests / yfinance / openai 之前
    `import ssl_fix`（本模块在被导入时自动打补丁，且幂等）。
"""

from __future__ import annotations

import platform
import ssl

_orig_load_default_certs = ssl.SSLContext.load_default_certs


def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    """容错版 load_default_certs：跳过损坏证书，必要时回退 certifi。"""
    try:
        # 先尝试原始逻辑（证书库正常时零开销）
        return _orig_load_default_certs(self, purpose)
    except ssl.SSLError:
        # 1) 首选 certifi 的 CA 包
        try:
            import certifi

            self.load_verify_locations(cafile=certifi.where())
            return None
        except Exception:  # noqa: BLE001 - certifi 缺失则走逐张加载
            pass

        # 2) 逐张加载 Windows 系统证书，跳过损坏的那张
        if platform.system() == "Windows":
            for storename in ("CA", "ROOT"):
                try:
                    for cert, encoding, trust in ssl.enum_certificates(storename):
                        if encoding != "x509_asn":
                            continue
                        if trust is True or purpose.oid in trust:
                            try:
                                self.load_verify_locations(cadata=cert)
                            except ssl.SSLError:
                                # 跳过这张损坏证书
                                continue
                except (PermissionError, OSError):
                    continue
        return None


def apply() -> bool:
    """应用补丁（幂等）。返回是否实际打了补丁。"""
    if platform.system() != "Windows":
        return False
    if getattr(ssl.SSLContext.load_default_certs, "_finagent_patched", False):
        return True
    _safe_load_default_certs._finagent_patched = True  # type: ignore[attr-defined]
    ssl.SSLContext.load_default_certs = _safe_load_default_certs  # type: ignore[assignment]
    return True


# 被 import 时自动生效
apply()
