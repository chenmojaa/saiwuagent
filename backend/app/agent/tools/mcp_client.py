"""Minimal stdio JSON-RPC 2.0 client for MCP servers.

We deliberately avoid pulling in `mcp` / `langchain-mcp-adapters`; the protocol
surface we need is tiny and predictable:

  - initialize / notifications/initialized
  - tools/list
  - tools/call

MCP stdio transport frames each JSON-RPC message as ONE line (newline
delimited) - NOT LSP-style Content-Length framing. Reading uses a daemon
thread + queue because ``select()`` does not work on Windows pipes.

Each call spawns the server fresh, performs the handshake, executes exactly one
request, then closes the process. Stdio MCP servers are idle between calls
so the overhead is acceptable; for hot paths callers should cache the process.

Failure modes: any non-zero return code, timeout, or unexpected JSON is
returned as a string for the LLM rather than raised; tool-call errors must
never crash the streaming loop.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

_CLIENT_VERSION = "0.1.0"
_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPServerSpec:
  server_id: str
  command: str
  args: list[str]
  env: dict[str, str]
  description: str = ""

  @classmethod
  def from_registry_entry(cls, entry: dict) -> "MCPServerSpec":
    return cls(
      server_id=str(entry.get("id") or entry.get("name") or "mcp"),
      command=str(entry.get("command") or ""),
      args=[str(a) for a in (entry.get("args") or [])],
      env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
      description=str(entry.get("description") or ""),
    )


def _reader_queue(proc: subprocess.Popen) -> "queue.Queue":
  """Return the process-wide reader queue (single daemon thread per process).

  One reader thread per process is REQUIRED: if every _read_message call
  spawned its own thread, the abandoned threads from earlier calls would
  keep consuming later responses from the same stream.
  """
  q = getattr(proc, "_mcp_reader_q", None)
  if q is None:
    q = queue.Queue()

    def _reader():
      try:
        for raw in proc.stdout:
          line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
          if not line:
            continue
          try:
            msg = json.loads(line)
          except ValueError:
            continue
          if isinstance(msg, dict):
            q.put(msg)
      except Exception:
        pass
      q.put(None)  # EOF / reader died

    threading.Thread(target=_reader, daemon=True).start()
    proc._mcp_reader_q = q  # type: ignore[attr-defined]
  return q


def _read_message(proc: subprocess.Popen, timeout: float) -> dict | None:
  """Read one JSON-RPC *response* (message with an ``id``) from the process.

  MCP stdio frames messages as newline-delimited JSON. Server notifications
  (no ``id``) and non-JSON stdout noise (npm banners etc.) are skipped.
  A daemon reader thread is used because ``select()`` cannot wait on
  Windows pipes.
  """
  q = _reader_queue(proc)
  deadline = time.time() + timeout
  while True:
    remaining = deadline - time.time()
    if remaining <= 0:
      return None
    try:
      msg = q.get(timeout=remaining)
    except queue.Empty:
      return None
    if msg is None:
      return None
    if "id" in msg:  # response to our request; skip notifications
      return msg


def _kill_tree(proc: subprocess.Popen) -> None:
  """Terminate the server AND its children (npx.cmd -> node)."""
  try:
    if os.name == "nt":
      subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                     capture_output=True,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
      proc.terminate()
  except Exception:
    pass


def _write_message(proc: subprocess.Popen, msg: dict) -> None:
  payload = json.dumps(msg, ensure_ascii=False).encode("utf-8")
  if proc.stdin is None:
    raise RuntimeError("subprocess stdin unavailable")
  proc.stdin.write(payload + b"\n")
  proc.stdin.flush()


def _spawn(spec: MCPServerSpec, cwd: str | None) -> subprocess.Popen:
  full_env = dict(os.environ)
  full_env.update(spec.env)
  creationflags = 0
  argv = [spec.command, *spec.args]
  if os.name == "nt":
    # Avoid showing console flashes for stdio MCP servers (npx, uvx, etc.).
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    # On Windows `npx` / `uvx` are .cmd shims; Popen(shell=False) cannot find
    # them by bare name, so resolve via PATH (PATHEXT-aware) first.
    cmd0 = argv[0]
    if os.path.sep not in cmd0 and not os.path.isfile(cmd0):
      resolved = shutil.which(cmd0)
      if resolved:
        argv[0] = resolved
      else:
        # Last resort: run through cmd /c so .cmd shims on PATH still work.
        argv = ["cmd", "/c", *argv]
  proc = subprocess.Popen(
    argv,
    cwd=cwd,
    env=full_env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=-1,
    creationflags=creationflags,
  )
  # MCP stdio servers tend to write to stderr for noisy logs. If we leave the
  # pipe full, the OS pipe buffer (~64 KiB on Windows / 64 KiB on POSIX) fills
  # and the child blocks on its next stderr write -> deadlock. Drain in a
  # background thread; only the most recent failure excerpt is retained for
  # surfacing in error messages.
  proc._stderr_tail: list[str] = []

  def _drain_stderr():
    if proc.stderr is None:
      return
    try:
      for raw in proc.stderr:
        try:
          line = raw.decode("utf-8", errors="replace")
        except Exception:
          line = repr(raw)
        proc._stderr_tail.append(line)
        if len(proc._stderr_tail) > 50:
          del proc._stderr_tail[:1]
        if len(proc._stderr_tail) > 50:
          del proc._stderr_tail[:1]
    except Exception:
      pass

  t = threading.Thread(target=_drain_stderr, name=f"mcp-stderr-{spec.server_id}", daemon=True)
  t.start()
  proc._stderr_thread = t
  return proc


def _handshake(proc: subprocess.Popen, init_timeout: float) -> bool:
  _write_message(proc, {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": _PROTOCOL_VERSION,
      "capabilities": {},
      "clientInfo": {"name": "small-warehouse-agent", "version": _CLIENT_VERSION},
    },
  })
  resp = _read_message(proc, init_timeout)
  if not resp or "result" not in resp:
    return False
  # notifications/initialized (no response expected).
  _write_message(proc, {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
    "params": {},
  })
  return True


def list_tools(spec: MCPServerSpec, init_timeout: float, call_timeout: float,
               cwd: str | None = None) -> list[dict]:
  """Return [{name, description, input_schema}, ...] for the given stdio MCP server."""
  try:
    proc = _spawn(spec, cwd)
  except (OSError, ValueError) as e:
    _log.warning("mcp %s: spawn failed: %s", spec.server_id, e)
    return []
  try:
    if not _handshake(proc, init_timeout):
      _log.warning("mcp %s: handshake failed", spec.server_id)
      return []
    _write_message(proc, {
      "jsonrpc": "2.0",
      "id": 2,
      "method": "tools/list",
      "params": {},
    })
    resp = _read_message(proc, call_timeout)
    if not resp:
      return []
    tools = ((resp.get("result") or {}).get("tools") or [])
    return [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)]
  finally:
    _kill_tree(proc)


def call_tool(spec: MCPServerSpec, tool_name: str, arguments: dict,
              init_timeout: float, call_timeout: float,
              cwd: str | None = None) -> str:
  """Invoke one tool on the stdio MCP server. Returns a string for the LLM."""
  try:
    proc = _spawn(spec, cwd)
  except (OSError, ValueError) as e:
    return "[mcp error] failed to start server %s: %s" % (spec.server_id, e)
  try:
    if not _handshake(proc, init_timeout):
      return "[mcp error] handshake failed for %s" % spec.server_id
    _write_message(proc, {
      "jsonrpc": "2.0",
      "id": 2,
      "method": "tools/call",
      "params": {"name": tool_name, "arguments": arguments or {}},
    })
    resp = _read_message(proc, call_timeout)
    if not resp:
      return "[mcp error] no response from %s within %.1fs" % (spec.server_id, call_timeout)
    if "error" in resp:
      err = resp["error"]
      return "[mcp error] %s: %s" % (err.get("code"), err.get("message") or err)
    result = resp.get("result") or {}
    content = result.get("content")
    if isinstance(content, list):
      parts: list[str] = []
      for item in content:
        if isinstance(item, dict):
          if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
          else:
            parts.append(json.dumps(item, ensure_ascii=False))
        elif isinstance(item, str):
          parts.append(item)
      return "\n".join(parts) or json.dumps(result, ensure_ascii=False)
    if isinstance(content, str):
      return content
    return json.dumps(result, ensure_ascii=False)
  finally:
    _kill_tree(proc)


class MCPSession:
  """Long-lived stdio MCP session: spawn + handshake ONCE, reuse for many calls.

  Rationale: spawning npx costs 2-3s per call (npx resolution + node boot +
  JSON-RPC handshake). A multi-step agent turn can invoke the same server
  5+ times, so keeping the process alive for the whole turn saves ~10s.

  Robustness: ``alive`` is checked before each call; a dead/EOF'd process is
  transparently respawned and the handshake redone, so callers never see the
  difference. ``close()`` kills the whole process tree (npx.cmd -> node).
  """

  def __init__(self, spec: MCPServerSpec, cwd: str | None = None,
               init_timeout: float = 10.0, call_timeout: float = 30.0):
    self.spec = spec
    self.cwd = cwd
    self.init_timeout = init_timeout
    self.call_timeout = call_timeout
    self._proc: subprocess.Popen | None = None
    self._next_id = 1  # 0 is reserved by JSON-RPC for request ids we skip
    self._calls = 0
    # 请求串行化。**这个类不是无状态可并发的**：_request() 是「写请求 ->
    # 读响应」，两步之间没有原子性保证。两个线程同时用同一个 session，
    # 响应会串线 —— A 的请求结果被 B 读走，而且**不会报错**，只是静默返回
    # 另一个调用的结果，比崩溃更难发现。
    #
    # 为什么必须在这里加而不是要求调用方加：会话是**池化共享**的
    # （mcp_tools._sessions / browser_search._session 都是模块级），
    # 两个并发请求会拿到同一个 session。指望每个调用方都记得加锁不现实。
    #
    # 用 RLock 而非 Lock：_ensure_started() 在握手失败时会调 self.close()，
    # 而它是在持锁状态下被调用的，需要可重入。
    self._lock = threading.RLock()

  # ---- lifecycle ----
  @property
  def alive(self) -> bool:
    return self._proc is not None and self._proc.poll() is None

  def _ensure_started(self) -> bool:
    if self.alive:
      return True
    try:
      self._proc = _spawn(self.spec, self.cwd)
    except (OSError, ValueError) as e:
      _log.warning("mcp %s: spawn failed: %s", self.spec.server_id, e)
      return False
    if not _handshake(self._proc, self.init_timeout):
      _log.warning("mcp %s: handshake failed", self.spec.server_id)
      self.close()
      return False
    self._next_id = 10  # handshake used id 1
    return True

  def close(self) -> None:
    # 持锁再杀进程：否则可能在另一个线程正读响应时把进程干掉。
    # 代价是 close() 会等在途请求结束（最多 call_timeout），但这正是我们要的
    # 语义 —— 宁可晚一点关，也不要让在途调用拿到半个响应。
    # RLock 可重入，_ensure_started() 在持锁状态下调到这里不会死锁。
    with self._lock:
      if self._proc is not None:
        _kill_tree(self._proc)
        self._proc = None

  # ---- protocol ----
  def _request(self, method: str, params: dict) -> dict | None:
    # 整个「写 + 读」必须原子：否则并发调用会串线（见 __init__ 里 _lock 的说明）。
    with self._lock:
      if not self._ensure_started():
        return None
      self._next_id += 1
      _write_message(self._proc, {
        "jsonrpc": "2.0",
        "id": self._next_id,
        "method": method,
        "params": params,
      })
      return _read_message(self._proc, self.call_timeout)

  def list_tools(self) -> list[dict]:
    resp = self._request("tools/list", {})
    if not resp:
      return []
    tools = ((resp.get("result") or {}).get("tools") or [])
    return [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)]

  def call(self, tool_name: str, arguments: dict) -> str:
    resp = self._request("tools/call", {"name": tool_name, "arguments": arguments or {}})
    self._calls += 1
    if resp is None:
      # Process likely died mid-turn; one transparent retry with fresh spawn.
      self.close()
      resp = self._request("tools/call", {"name": tool_name, "arguments": arguments or {}})
    if resp is None:
      return "[mcp error] no response from %s within %.1fs" % (self.spec.server_id, self.call_timeout)
    if "error" in resp:
      err = resp["error"]
      return "[mcp error] %s: %s" % (err.get("code"), err.get("message") or err)
    result = resp.get("result") or {}
    content = result.get("content")
    if isinstance(content, list):
      parts: list[str] = []
      for item in content:
        if isinstance(item, dict):
          if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
          else:
            parts.append(json.dumps(item, ensure_ascii=False))
        elif isinstance(item, str):
          parts.append(item)
      return "\n".join(parts) or json.dumps(result, ensure_ascii=False)
    if isinstance(content, str):
      return content
    return json.dumps(result, ensure_ascii=False)

  @property
  def calls(self) -> int:
    return self._calls