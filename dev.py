#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HD 一键开发脚本（跨平台 Windows / macOS / Linux）

用法：
    python dev.py              # 启动前后端（首次自动装依赖）
    python dev.py setup        # 只安装前后端依赖
    python dev.py backend      # 只启动后端（前台，日志直出终端）
    python dev.py frontend     # 只启动前端（前台，日志直出终端）
    python dev.py --no-setup   # 跳过依赖检查，直接启动

设计目标：
    别人 git clone 之后，只要本机有 Python 3.10+ 和 Node.js 18+，
    跑一条 `python dev.py` 就能把前后端都拉起来。
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

IS_WINDOWS = os.name == "nt"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5006  # 与 frontend/vite.config.ts 的 /api 代理默认值保持一致
FRONTEND_PORT = 5174

# backend/.venv 内标记文件：装完依赖后写入，用于判断是否需要重装
_SETUP_MARK = ".hd_setup_done"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def log(msg: str):
    print(f"[dev] {msg}", flush=True)


def warn(msg: str):
    print(f"[dev] \u26a0 {msg}", flush=True)


def venv_python() -> Path:
    if IS_WINDOWS:
        return BACKEND / ".venv" / "Scripts" / "python.exe"
    return BACKEND / ".venv" / "bin" / "python"


def read_env() -> tuple[str, int]:
    """从 backend/.env 读取 HOST / PORT，缺失时用默认值。"""
    env_file = BACKEND / ".env"
    host, port = DEFAULT_HOST, DEFAULT_PORT
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "HOST" and value:
                host = value
            elif key == "PORT" and value:
                try:
                    port = int(value)
                except ValueError:
                    pass
    return host, port


def detect_package_manager() -> str | None:
    """优先 pnpm，其次 npm。返回可执行文件的**完整路径**。

    Windows 上 pnpm / npm 都是 .CMD 包装脚本，而 CreateProcess 不像 cmd.exe
    那样自动补 .cmd 扩展名——只传裸名字 "pnpm" 会直接抛
    FileNotFoundError: [WinError 2]。实测后果：后端已经拉起来了，前端 spawn
    失败把整个 dev.py 带崩，且此时还没进 try 块，后端子进程成了孤儿。
    所以这里必须返回 shutil.which() 的完整路径。
    """
    for pm in ("pnpm", "npm"):
        found = shutil.which(pm)
        if found:
            return found
    return None


def run(cmd: list[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess:
    """前台运行一个命令，日志直出终端。"""
    log("> " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check)


# --------------------------------------------------------------------------- #
# 依赖安装
# --------------------------------------------------------------------------- #
def ensure_backend_env() -> None:
    """确保 .env 存在，不存在则从 .env.example 复制。"""
    env_file = BACKEND / ".env"
    example = BACKEND / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copyfile(example, env_file)
        log("已生成 backend/.env（来自 .env.example），请填入 LLM_API_KEY 等密钥")


def setup_backend(force: bool = False) -> None:
    ensure_backend_env()
    py = venv_python()
    mark = py.parent.parent / _SETUP_MARK  # backend/.venv/.hd_setup_done

    if py.exists() and mark.exists() and not force:
        log("后端依赖已安装（backend/.venv），跳过。用 `python dev.py setup` 强制重装")
        return

    if not py.exists():
        log("创建 Python 虚拟环境 backend/.venv ...")
        run([sys.executable, "-m", "venv", str(BACKEND / ".venv")], cwd=ROOT)

    log("安装后端依赖（pip install -r requirements.txt）...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], cwd=BACKEND)
    run([str(py), "-m", "pip", "install", "-r", "requirements.txt"], cwd=BACKEND)
    mark.write_text("ok", encoding="utf-8")
    log("后端依赖安装完成")


def setup_frontend(force: bool = False) -> None:
    node_modules = FRONTEND / "node_modules"
    if node_modules.exists() and not force:
        log("前端依赖已安装（frontend/node_modules），跳过。用 `python dev.py setup` 强制重装")
        return

    pm = detect_package_manager()
    if pm is None:
        warn("未检测到 pnpm / npm。请先安装 Node.js 18+：https://nodejs.org")
        return

    log(f"安装前端依赖（{pm} install）...")
    run([pm, "install"], cwd=FRONTEND)
    log("前端依赖安装完成")


def do_setup(force: bool = True) -> None:
    setup_backend(force=force)
    setup_frontend(force=force)


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def _spawn(cmd: list[str], cwd: Path, logfile: Path) -> subprocess.Popen:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    f = open(logfile, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT
    )


def wait_http_ready(url: str, timeout: int = 30) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def stop_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is None:
        log(f"正在停止 {name} ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def start_backend_foreground() -> None:
    host, port = read_env()
    py = venv_python()
    if not py.exists():
        log("未找到 backend/.venv，先执行 `python dev.py setup`")
        sys.exit(1)
    cmd = [str(py), "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]
    log(f"启动后端 FastAPI -> http://{host}:{port}")
    subprocess.run(cmd, cwd=str(BACKEND))


def start_frontend_foreground() -> None:
    pm = detect_package_manager()
    if pm is None:
        warn("未检测到 pnpm / npm，请先安装 Node.js 18+")
        sys.exit(1)
    cmd = [pm, "run", "dev"]
    log(f"启动前端 Vite -> http://127.0.0.1:{FRONTEND_PORT}")
    subprocess.run(cmd, cwd=str(FRONTEND))


def start_both() -> None:
    host, port = read_env()
    py = venv_python()

    procs: list[tuple[str, subprocess.Popen]] = []

    # 后端
    backend_log = BACKEND / "uvicorn.log"
    cmd_backend = [
        str(py), "-m", "uvicorn", "app.main:app",
        "--host", host, "--port", str(port),
    ]
    log(f"启动后端 FastAPI（日志: backend/uvicorn.log）-> http://{host}:{port}")
    bp = _spawn(cmd_backend, BACKEND, backend_log)
    procs.append(("后端", bp))

    if not wait_http_ready(f"http://127.0.0.1:{port}/api/health"):
        warn(f"后端 {port} 端口未在 {30}s 内就绪，请检查 backend/uvicorn.log")

    # 前端
    pm = detect_package_manager()
    if pm is None:
        warn("未检测到 pnpm / npm，仅启动了后端")
    else:
        frontend_log = FRONTEND / "vite.log"
        cmd_frontend = [pm, "run", "dev"]
        log(f"启动前端 Vite（日志: frontend/vite.log）-> http://127.0.0.1:{FRONTEND_PORT}")
        fp = _spawn(cmd_frontend, FRONTEND, frontend_log)
        procs.append(("前端", fp))
        wait_http_ready(f"http://127.0.0.1:{FRONTEND_PORT}/", timeout=15)

    print("", flush=True)
    print(f"  前端入口  : http://127.0.0.1:{FRONTEND_PORT}", flush=True)
    print(f"  后端健康  : http://127.0.0.1:{port}/api/health", flush=True)
    print("  按 Ctrl+C 同时停止前后端", flush=True)
    print("", flush=True)

    try:
        while True:
            time.sleep(1)
            # 任一进程提前退出则终止另一个
            for name, p in procs:
                if p.poll() is not None:
                    log(f"{name} 已退出（code={p.poll()}），正在停止其余进程")
                    for n2, p2 in procs:
                        stop_process(p2, n2)
                    sys.exit(p.returncode or 1)
    except KeyboardInterrupt:
        log("收到退出信号，正在关闭前后端 ...")
        for name, p in procs:
            stop_process(p, name)
        log("已全部停止")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    args = sys.argv[1:]
    command = None
    no_setup = "--no-setup" in args
    if args and not args[0].startswith("--"):
        command = args[0].lower()

    if command == "setup":
        do_setup(force=True)
        return
    if command == "backend":
        if not no_setup:
            setup_backend()
        start_backend_foreground()
        return
    if command == "frontend":
        if not no_setup:
            setup_frontend()
        start_frontend_foreground()
        return

    # 默认：启动前后端
    if not no_setup:
        setup_backend()
        setup_frontend()
    start_both()


if __name__ == "__main__":
    main()
