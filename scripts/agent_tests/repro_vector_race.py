# -*- coding: utf-8 -*-
"""复现 vector.get_collection() 的懒加载竞态。

现象：parallel_plan_node 用 ThreadPoolExecutor 并发跑多个 hybrid_search，
多线程同时首次调用 get_collection() 时会各自创建一个 PersistentClient，
ChromaDB 报 "Could not connect to tenant default_tenant"。

本脚本把模块级单例清空后，用多线程同时触发首次初始化，复现该错误。
"""
import sys
import threading
import traceback
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))
import os
os.chdir(_BACKEND)

from app.storage import vector  # noqa: E402

ROUNDS = 5
THREADS = 8

print("=" * 66)
print("  复现 vector.get_collection() 并发初始化竞态")
print("  每轮: 清空单例 -> %d 线程同时首次调用" % THREADS)
print("=" * 66)

errors: list[str] = []
ok = 0

for rnd in range(1, ROUNDS + 1):
    # 强制回到"未初始化"状态
    vector._client = None
    vector._collection = None

    barrier = threading.Barrier(THREADS)
    local_err: list[str] = []
    local_ok = [0]

    def worker():
        try:
            barrier.wait(timeout=10)      # 让所有线程同时冲
            col = vector.get_collection()
            if col is not None:
                local_ok[0] += 1
        except Exception as e:
            local_err.append("%s: %s" % (type(e).__name__, str(e)[:120]))

    ts = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)

    ok += local_ok[0]
    errors.extend(local_err)
    status = "OK  " if not local_err else "RACE"
    print("  第 %d 轮 [%s] 成功 %d/%d  错误 %d" % (
        rnd, status, local_ok[0], THREADS, len(local_err)))
    for e in local_err[:3]:
        print("        -> %s" % e)

print("-" * 66)
print("  成功 %d / 总尝试 %d，捕获 %d 个错误" % (ok, ROUNDS * THREADS, len(errors)))
if errors:
    print("\n  去重后的错误类型：")
    for e in sorted(set(errors))[:6]:
        print("    - %s" % e)
    print("\n  结论：get_collection() 缺少初始化锁，并发首次调用会炸。")
    sys.exit(1)
else:
    print("\n  未复现（可能 ChromaDB 版本已内部加锁）")
    sys.exit(0)
