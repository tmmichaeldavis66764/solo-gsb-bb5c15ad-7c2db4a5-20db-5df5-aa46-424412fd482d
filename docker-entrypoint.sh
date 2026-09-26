#!/bin/sh
# 容器一键启动入口:
# 1) 依赖初始化纳入启动 —— 首次启动(或镜像内缺依赖时)自动安装 requirements.txt,
#    已安装则秒过; 镜像构建阶段也会预装, 离线/正常环境均可用同一条命令启动。
# 2) 数据库表结构初始化随 FastAPI startup 自动完成(app/main.py:init_db),
#    启动前先确保数据目录存在。
set -e

REQ_FILE="/app/requirements.txt"

need_install=0
python3 - <<'PY' 2>/dev/null || need_install=1
import importlib
for mod in ("fastapi", "uvicorn", "pydantic"):
    importlib.import_module(mod)
PY

if [ "$need_install" -eq 1 ] && [ -f "$REQ_FILE" ]; then
    echo "[entrypoint] 首次启动: 正在安装 Python 依赖..."
    python3 -m pip install --no-cache-dir -r "$REQ_FILE"
fi

mkdir -p "$(dirname "${BALANCER_DB:-/app/data/balancer.db}")"

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
