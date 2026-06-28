"""启动入口：先修复 Windows 证书库 SSL 报错，再启动 Streamlit 应用。

用法（替代 `streamlit run frontend/app.py`）：
    python run_app.py

为什么需要它：
    `streamlit run ...` 在导入 tornado 时就会调用 ssl.create_default_context()，
    若系统证书库存在损坏证书会直接崩溃，根本进不到 app.py。本入口在导入
    streamlit / tornado 之前先 `import ssl_fix` 打上补丁，从根上规避该问题，
    同时也覆盖运行时 yfinance / DeepSeek / SEC 的 HTTPS 请求。
"""

import sys
from pathlib import Path

# 关键：必须在导入 streamlit / tornado 之前应用 SSL 补丁
import ssl_fix  # noqa: F401,E402

from streamlit.web import cli as stcli  # noqa: E402


def main() -> None:
    app_path = str(Path(__file__).resolve().parent / "frontend" / "app.py")
    # 透传用户在命令行额外附加的 streamlit 参数（如 --server.port 8502）
    extra_args = sys.argv[1:]
    sys.argv = ["streamlit", "run", app_path, *extra_args]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
